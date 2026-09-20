"""
ablation.py
-----------
FIGNN architecture ablation.

Compares the historical smaller architecture:
    hidden_dim = 64
    n_heads    = 4
    latent_dim = 32

against the final architecture:
    hidden_dim = 128
    n_heads    = 8
    latent_dim = 64

IMPORTANT:
    src/models/model.py is NOT modified.

The ablation reproduces the Stage-1 LOSO training setup:
    - COG-BCI, REST removed
    - strict subject-wise LOSO
    - training-subject normalization statistics
    - random 85/15 graph-level train/validation split
    - WeightedRandomSampler
    - SupConLoss, temperature=0.1
    - AdamW, lr=1e-3, weight_decay=1e-4
    - CosineAnnealingLR, eta_min=1e-5
    - 500 epochs
    - best checkpoint selected by validation silhouette

Initial run:
    5 folds

After validating the implementation, change:
    ABLATION_SUBJECTS = list(range(1, 30))
for the full 29-fold run.
"""

import os
import sys
import pickle
import subprocess

import torch
import torch.nn as nn
import numpy as np

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_metric_learning.losses import SupConLoss


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "/home/nvidia/24BCE1793/fignn_cni__project"

DATA_PATH = (
    "/lp-dev/24BCE1793/data/processed/"
    "all_graphs_29sub.pkl"
)

SAVE_ROOT = (
    "/lp-dev/24BCE1793/outputs/"
    "ablation_64_4_32"
)


# ============================================================
# ABLATION SETTINGS
# ============================================================

# FULL 29-FOLD ARCHITECTURE SENSITIVITY RUN
ABLATION_SUBJECTS = list(range(1, 30))


# Same as final LOSO training
N_EPOCHS = 500
BATCH_SIZE = 128
LR = 1e-3
VAL_SPLIT = 0.15


# Historical architecture
ABLATION_HIDDEN_DIM = 64
ABLATION_N_HEADS = 4
ABLATION_LATENT_DIM = 32

DROPOUT = 0.3


# Four GPUs available on Brev. One fold per GPU at a time.
# Subjects are scheduled in groups of four. Each worker process is
# restricted to exactly one physical GPU via CUDA_VISIBLE_DEVICES.
GPUS = [1, 2, 5, 6]


os.makedirs(SAVE_ROOT, exist_ok=True)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(0, PROJECT_DIR)

from src.models.model import FIGNNEncoder


# ============================================================
# DATASET NORMALISATION
# ============================================================

def compute_dataset_stats(graph_dataset):

    print(
        "Computing dataset feature statistics...",
        flush=True
    )

    all_x = torch.cat(
        [g.x for g in graph_dataset],
        dim=0
    )

    mean = all_x.mean(dim=0)

    std = all_x.std(
        dim=0
    ).clamp(min=1e-8)

    print(
        f"  Feature mean: {mean.tolist()}",
        flush=True
    )

    print(
        f"  Feature std:  {std.tolist()}",
        flush=True
    )

    return mean, std


def normalise_dataset(graph_dataset, mean, std):

    print(
        "Normalising node features (z-score, no log)...",
        flush=True
    )

    for g in graph_dataset:
        g.x = (g.x - mean) / std

    print(
        "Done.",
        flush=True
    )


# ============================================================
# BALANCED SAMPLER
# ============================================================

def make_balanced_sampler(dataset):

    labels = [
        g.y.item()
        for g in dataset
    ]

    class_counts = torch.bincount(
        torch.tensor(labels)
    )

    weights = [
        1.0 / class_counts[l].item()
        for l in labels
    ]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


# ============================================================
# TRAINING
# ============================================================

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

    return total_loss / max(n_batches, 1)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    all_z = []
    all_labels = []

    for batch in loader:

        batch = batch.to(device)

        z = (
            model(batch)
            .cpu()
            .numpy()
        )

        all_z.append(z)

        all_labels.append(
            batch.y
            .squeeze()
            .cpu()
            .numpy()
        )

    return (
        np.vstack(all_z),
        np.concatenate(all_labels)
    )


# ============================================================
# LOCAL ABLATION TRAINER
# ============================================================

def run_ablation_training(
    train_graphs,
    test_graphs,
    save_dir
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}",
        flush=True
    )

    print(
        "\nABLATION ARCHITECTURE",
        flush=True
    )

    print(
        "hidden_dim = 64",
        flush=True
    )

    print(
        "n_heads    = 4",
        flush=True
    )

    print(
        "latent_dim = 32",
        flush=True
    )

    # --------------------------------------------------------
    # 1. Normalisation
    # --------------------------------------------------------

    mean, std = compute_dataset_stats(
        train_graphs
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

    normalise_dataset(
        train_graphs,
        mean,
        std
    )

    normalise_dataset(
        test_graphs,
        mean,
        std
    )

    # --------------------------------------------------------
    # 2. Random train / validation split
    # --------------------------------------------------------

    n_total = len(train_graphs)

    n_val = int(
        n_total * VAL_SPLIT
    )

    indices = torch.randperm(
        n_total
    ).tolist()

    train_set = [
        train_graphs[i]
        for i in indices[:-n_val]
    ]

    val_set = [
        train_graphs[i]
        for i in indices[-n_val:]
    ]

    # --------------------------------------------------------
    # 3. DataLoaders
    # --------------------------------------------------------

    sampler = make_balanced_sampler(
        train_set
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    sample = next(
        iter(train_loader)
    )

    print(
        f"Sample batch labels: "
        f"{torch.bincount(sample.y.squeeze())}",
        flush=True
    )

    print(
        f"Train: {len(train_set)} graphs",
        flush=True
    )

    print(
        f"Val:   {len(val_set)} graphs",
        flush=True
    )

    print(
        f"Test:  {len(test_graphs)} graphs",
        flush=True
    )

    # --------------------------------------------------------
    # 4. MODEL
    # --------------------------------------------------------

    model = FIGNNEncoder(
        hidden_dim=ABLATION_HIDDEN_DIM,
        latent_dim=ABLATION_LATENT_DIM,
        n_heads=ABLATION_N_HEADS,
        dropout=DROPOUT
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {n_params:,}",
        flush=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-5
    )

    loss_fn = SupConLoss(
        temperature=0.1
    )

    # --------------------------------------------------------
    # 5. TRAINING
    # --------------------------------------------------------

    best_val_loss = float("inf")

    train_history = []
    val_history = []

    for epoch in range(
        1,
        N_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        scheduler.step()

        train_history.append(
            train_loss
        )

        if epoch % 10 == 0 or epoch == 1:
        
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
        
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    z = model(batch)
                    val_loss = loss_fn(z, batch.y.squeeze())
                    val_loss_total += val_loss.item()
                    val_batches += 1
        
            val_loss = val_loss_total / max(val_batches, 1)
            val_history.append(val_loss)
        
            print(
                f"Epoch {epoch:>3}/{N_EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                flush=True
            )
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_dir,
                        "fignn_best.pth"
                    )
                )
        else:

            print(
                f"Epoch {epoch:>3}/{N_EPOCHS}  "
                f"train_loss={train_loss:.4f}",
                flush=True
            )

    # --------------------------------------------------------
    # 6. TRAINING SUMMARY
    # --------------------------------------------------------

    print(
        f"\nBest val SupCon loss: "
        f"{best_val_loss:.4f}",
        flush=True
    )

    print(
        f"Band importances: "
        f"{model.get_band_importance()}",
        flush=True
    )

    # --------------------------------------------------------
    # 7. REQUIRED ABLATION ARTIFACTS ONLY
    # --------------------------------------------------------
    #
    # Keep only:
    #   fignn_best.pth
    #   normalisation_stats.pt
    #   results.txt (written by run_fold)
    #
    # We intentionally do NOT save fignn_final.pth,
    # test_z.npy, test_labels.npy, train_history.npy, or
    # val_history.npy. The validation notebook reloads the
    # best checkpoint and recomputes embeddings, so these
    # extra files are unnecessary for the 29-fold ablation.
    # --------------------------------------------------------

    return {
        "best_val_loss": float(best_val_loss),
        "n_parameters": int(n_params),
        "latent_dim": ABLATION_LATENT_DIM
    }

# ============================================================
# ONE LOSO FOLD
# ============================================================

def run_fold(
    subject,
    gpu
):

    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = str(gpu)

    sys.path.insert(
        0,
        PROJECT_DIR
    )

    os.chdir(
        PROJECT_DIR
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        f"ABLATION LOSO FOLD: "
        f"SUBJECT {subject:02d}",
        flush=True
    )

    print(
        f"GPU: {gpu}",
        flush=True
    )

    print(
        "Architecture: 64 hidden / "
        "4 heads / 32 latent",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    with open(
        DATA_PATH,
        "rb"
    ) as f:

        graphs = pickle.load(f)

    # Same Stage-1 setup:
    # REST = label 3, so remove it.
    graphs = [
        g
        for g in graphs
        if g.y.item() != 3
    ]

    # Strict LOSO
    train_graphs = [
        g
        for g in graphs
        if int(g.subject) != subject
    ]

    test_graphs = [
        g
        for g in graphs
        if int(g.subject) == subject
    ]

    train_subjects = sorted(
        set(int(g.subject) for g in train_graphs)
    )
    
    print(
        f"Train subjects: {train_subjects}",
        flush=True
    )

    print(
        f"Held-out subject: {subject}",
        flush=True
    )

    print(
        f"Train graphs: "
        f"{len(train_graphs)}",
        flush=True
    )

    print(
        f"Test graphs: "
        f"{len(test_graphs)}",
        flush=True
    )

    fold_dir = os.path.join(
        SAVE_ROOT,
        f"subject_{subject:02d}"
    )

    os.makedirs(
        fold_dir,
        exist_ok=True
    )

    results = run_ablation_training(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        save_dir=fold_dir
    )

    # Save simple result file
    result_path = os.path.join(
        fold_dir,
        "results.txt"
    )

    with open(
        result_path,
        "w"
    ) as f:

        f.write(
            f"subject={subject}\n"
        )

        f.write(
            "hidden_dim=64\n"
        )

        f.write(
            "n_heads=4\n"
        )

        f.write(
            "latent_dim=32\n"
        )

        f.write(
            f"n_parameters="
            f"{results['n_parameters']}\n"
        )

        f.write(
            f"best_val_loss="
            f"{results['best_val_loss']}\n"
        )


    print(
        f"\nSUBJECT {subject:02d} COMPLETE",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    remaining = ABLATION_SUBJECTS.copy()

    print(
        "\n" + "=" * 70
    )

    print(
        "FIGNN ARCHITECTURE ABLATION — FULL 29 FOLDS"
    )

    print(
        "Architecture: 64 hidden / 4 heads / 32 latent"
    )

    print(
        f"Subjects: {ABLATION_SUBJECTS}"
    )

    print(
        f"Epochs: {N_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"GPUs: {GPUS}"
    )

    print(
        "=" * 70 + "\n"
    )

    while remaining:

        current = []

        for gpu in GPUS:

            if not remaining:
                break

            subject = remaining.pop(0)

            current.append(
                (subject, gpu)
            )

        processes = []

        for subject, gpu in current:

            print(
                f"Launching Subject "
                f"{subject:02d} on GPU {gpu}",
                flush=True
            )

            log_path = os.path.join(
                SAVE_ROOT,
                f"subject_{subject:02d}.log"
            )

            log_file = open(
                log_path,
                "w"
            )

            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            p = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    str(subject),
                    str(gpu)
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=worker_env
            )

            processes.append(
                (
                    subject,
                    p,
                    log_file
                )
            )

        # Wait for this group
        for (
            subject,
            p,
            log_file
        ) in processes:

            code = p.wait()

            log_file.close()

            if code != 0:

                print(
                    f"ERROR: Subject "
                    f"{subject:02d} failed "
                    f"with code {code}",
                    flush=True
                )

            else:

                print(
                    f"Subject "
                    f"{subject:02d} finished.",
                    flush=True
                )

    print(
        "\n" + "=" * 70
    )

    print(
        "ABLATION RUN COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if "--worker" in sys.argv:

        subject = int(
            sys.argv[2]
        )

        gpu = int(
            sys.argv[3]
        )

        run_fold(
            subject,
            gpu
        )

    else:

        main()