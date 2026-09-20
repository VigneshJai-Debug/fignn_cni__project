"""
train.py
--------
Training loop for FIGNN Stage 1.

The script handles:
    - Subject-wise train/validation splitting
    - Training-only normalization
    - Balanced graph sampling
    - SupCon training
    - Validation-based checkpoint selection

Evaluation, LOSO metrics, q_s, binomial tests,
and threshold sensitivity are handled separately in validation.ipynb.
"""

import os
import random
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_metric_learning.losses import SupConLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_subject_id(graph):
    """
    Return the subject ID stored on each graph.

    Update this list if the graph builder uses another attribute name.
    """
    for name in [
        "subject_id",
        "subject",
        "participant",
        "participant_id",
        "subj",
        "sid"
    ]:
        if hasattr(graph, name):
            value = getattr(graph, name)

            if torch.is_tensor(value):
                value = value.item()

            return value

    raise AttributeError(
        "Subject ID not found on graph. "
        "Store subject metadata on every graph before training."
    )


def make_subjectwise_split(graphs, val_split=0.15, seed=42):
    """
    Split graphs by subject.

    All graphs from one subject remain in the same partition.
    """
    subject_to_graphs = {}

    for graph in graphs:
        subject_id = get_subject_id(graph)
        subject_to_graphs.setdefault(subject_id, []).append(graph)

    # Explicit sorting ensures the same subject split across runs
    # regardless of graph insertion order.
    subjects = sorted(subject_to_graphs.keys())

    if len(subjects) < 2:
        raise ValueError(
            "Subject-wise validation requires at least two subjects."
        )

    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val_subjects = max(
        1,
        int(round(len(subjects) * val_split))
    )

    n_val_subjects = min(
        n_val_subjects,
        len(subjects) - 1
    )

    val_subjects = set(subjects[:n_val_subjects])
    train_subjects = set(subjects[n_val_subjects:])

    train_set = [
        graph
        for subject_id in train_subjects
        for graph in subject_to_graphs[subject_id]
    ]

    val_set = [
        graph
        for subject_id in val_subjects
        for graph in subject_to_graphs[subject_id]
    ]

    assert train_subjects.isdisjoint(val_subjects)

    print(
        f"Training subjects: {sorted(train_subjects, key=str)}",
        flush=True
    )

    print(
        f"Validation subjects: {sorted(val_subjects, key=str)}",
        flush=True
    )

    print(
        f"Training graphs: {len(train_set)} | "
        f"Validation graphs: {len(val_set)}",
        flush=True
    )

    return train_set, val_set


def compute_dataset_stats(graph_dataset):
    """
    Compute normalization statistics using training graphs only.
    """
    print(
        "Computing normalization statistics from training graphs...",
        flush=True
    )

    all_x = torch.cat(
        [graph.x for graph in graph_dataset],
        dim=0
    )

    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0).clamp(min=1e-8)

    print(
        f"Feature mean: {mean.tolist()}",
        flush=True
    )

    print(
        f"Feature std:  {std.tolist()}",
        flush=True
    )

    return mean, std


def normalise_dataset(graph_dataset, mean, std):
    """
    Apply the training-subject normalization statistics.
    """
    for graph in graph_dataset:
        graph.x = (graph.x - mean) / std


def make_balanced_sampler(dataset):
    """
    Create a weighted sampler to reduce class imbalance.
    """
    labels = [
        int(graph.y.item())
        for graph in dataset
    ]

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    class_counts = torch.bincount(labels)

    weights = [
        1.0 / class_counts[label].item()
        for label in labels.tolist()
    ]

    return WeightedRandomSampler(
        weights=torch.tensor(
            weights,
            dtype=torch.double
        ),
        num_samples=len(weights),
        replacement=True
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device
):
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        z = model(batch)
        labels = batch.y.squeeze()

        loss = loss_fn(
            z,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(
        n_batches,
        1
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device
):
    """
    Compute SupCon loss on the subject-wise
    validation set without updating model parameters.
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        z = model(batch)
        labels = batch.y.squeeze()

        loss = loss_fn(
            z,
            labels
        )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(
        n_batches,
        1
    )


def run_training(
    train_graphs,
    test_graphs=None,
    save_dir="outputs/checkpoints",
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    val_split=0.15,
    seed=42
):
    """
    Train FIGNN using a subject-wise validation split.

    The best checkpoint is selected using the lowest
    validation SupCon loss.

    The held-out LOSO test subject is never used for
    training or checkpoint selection.
    """
    from src.models.model import (
        FIGNNEncoder,
        count_parameters
    )

    set_seed(seed)

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}",
        flush=True
    )

    # -------------------------------------------------------------
    # 1. Subject-wise train/validation split
    # -------------------------------------------------------------

    train_set, val_set = make_subjectwise_split(
        train_graphs,
        val_split=val_split,
        seed=seed
    )

    # -------------------------------------------------------------
    # 2. Training-only normalization
    # -------------------------------------------------------------

    mean, std = compute_dataset_stats(
        train_set
    )

    torch.save(
        {
            "mean": mean.cpu(),
            "std": std.cpu()
        },
        os.path.join(
            save_dir,
            "normalisation_stats.pt"
        )
    )

    # Apply statistics computed ONLY from training subjects
    # to train, validation, and held-out test graphs.
    normalise_dataset(
        train_set,
        mean,
        std
    )

    normalise_dataset(
        val_set,
        mean,
        std
    )

    if test_graphs is not None:
        normalise_dataset(
            test_graphs,
            mean,
            std
        )

    # -------------------------------------------------------------
    # 3. DataLoaders
    # -------------------------------------------------------------

    sampler = make_balanced_sampler(
        train_set
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(
        f"Train: {len(train_set)} graphs | "
        f"Val: {len(val_set)} graphs",
        flush=True
    )

    # -------------------------------------------------------------
    # 4. Model and optimizer
    # -------------------------------------------------------------

    model = FIGNNEncoder().to(device)

    count_parameters(
        model
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs,
        eta_min=1e-5
    )

    loss_fn = SupConLoss(
        temperature=0.1
    )

    # -------------------------------------------------------------
    # 5. Training
    # -------------------------------------------------------------

    train_history = []

    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(
        1,
        n_epochs + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        scheduler.step()

        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Epoch {epoch:>3}/{n_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}",
            flush=True
        )

        # ---------------------------------------------------------
        # Best checkpoint
        # Lowest validation SupCon loss
        # ---------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    "fignn_best.pth"
                )
            )

    # -------------------------------------------------------------
    # 6. Save final model
    # -------------------------------------------------------------

    torch.save(
        model.state_dict(),
        os.path.join(
            save_dir,
            "fignn_final.pth"
        )
    )

    print(
        "\nTraining complete.",
        flush=True
    )

    print(
        f"Best validation checkpoint: "
        f"epoch {best_epoch}, "
        f"val_loss={best_val_loss:.6f}",
        flush=True
    )

    print(
        f"Final epoch: {n_epochs}",
        flush=True
    )

    print(
        f"Band importances: "
        f"{model.get_band_importance()}",
        flush=True
    )

    return model, train_history