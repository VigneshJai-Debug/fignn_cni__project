"""
train.py
--------
Training loop for FIGNN Stage 1.

Loss: SupConLoss
  For each batch, all same-class pairs are pulled together
  and all different-class pairs are pushed apart simultaneously.
  More gradient signal per batch than triplet mining.

Why not cross-entropy:
  We want a meaningful geometry in z-space that generalises to
  unseen subjects. Cross-entropy overfits to training subjects.
  Contrastive loss learns structure that holds across people.

Training epochs vs EEG epochs:
  EEG epochs  = 2-second windows = graph samples (~43,550 total)
  Training epochs = full passes through the dataset (100 of these)
  One training epoch = model sees all 43,550 graphs once.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_metric_learning.losses import SupConLoss
import numpy as np
import os


def compute_dataset_stats(graph_dataset):
    print("Computing dataset feature statistics...", flush=True)
    all_x = torch.cat([g.x for g in graph_dataset], dim=0)  # (N, 4)
    # Relative band power is already in [0,1] — just z-score normalise
    # No log transform needed since values are already proportions
    mean = all_x.mean(dim=0)   # (4,)
    std  = all_x.std(dim=0).clamp(min=1e-8)   # (4,)
    print(f"  Feature mean: {mean.tolist()}", flush=True)
    print(f"  Feature std:  {std.tolist()}", flush=True)
    return mean, std


def normalise_dataset(graph_dataset, mean, std):
    print("Normalising node features (z-score, no log)...", flush=True)
    for g in graph_dataset:
        g.x = (g.x - mean) / std
    print("Done.", flush=True)


def make_balanced_sampler(dataset):
    """
    Returns a WeightedRandomSampler that draws equal numbers from each class.
    Guarantees every batch contains all 3 cognitive load levels.
    """
    labels       = [g.y.item() for g in dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    weights      = [1.0 / class_counts[l].item() for l in labels]
    return WeightedRandomSampler(
        weights    = weights,
        num_samples= len(weights),
        replacement= True
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    n_batches  = 0

    for batch in loader:
        batch  = batch.to(device)
        optimizer.zero_grad()

        z      = model(batch)
        labels = batch.y.squeeze()

        loss   = loss_fn(z, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Collect all z vectors and labels for validation / t-SNE.
    Returns numpy arrays: z (N, 32) and labels (N,).
    """
    model.eval()
    all_z      = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        z     = model(batch).cpu().numpy()
        all_z.append(z)
        all_labels.append(batch.y.squeeze().cpu().numpy())

    return np.vstack(all_z), np.concatenate(all_labels)


def run_training(graph_dataset: list,
                 save_dir  : str   = "outputs/checkpoints",
                 n_epochs  : int   = 100,
                 batch_size: int   = 64,
                 lr        : float = 1e-3,
                 val_split : float = 0.15):
    """
    Full training run.

    Args:
        graph_dataset : list of PyG Data objects (all subjects, all tasks)
        save_dir      : where to save model weights
        n_epochs      : training epochs
        batch_size    : graphs per batch
        lr            : learning rate
        val_split     : fraction held out for validation
    """
    from src.models.model import FIGNNEncoder, count_parameters

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ── 1. Normalise node features across full dataset ──────────────────
    mean, std = compute_dataset_stats(graph_dataset)
    normalise_dataset(graph_dataset, mean, std)
    # quick check after normalising
    check = torch.cat([g.x for g in graph_dataset[:1000]], dim=0)
    print(f"Post-norm x mean: {check.mean(dim=0).tolist()}", flush=True)
    print(f"Post-norm x std:  {check.std(dim=0).tolist()}", flush=True)

    # ── 2. Train / val split ────────────────────────────────────────────
    n_total = len(graph_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    indices   = torch.randperm(n_total).tolist()
    train_set = [graph_dataset[i] for i in indices[:n_train]]
    val_set   = [graph_dataset[i] for i in indices[n_train:]]

    # ── 3. DataLoaders ──────────────────────────────────────────────────
    sampler      = make_balanced_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Sanity check — confirm all 3 labels appear in a sample batch
    sample = next(iter(train_loader))
    print(f"Sample batch labels: {torch.bincount(sample.y.squeeze())}", flush=True)
    print(f"Train: {len(train_set)} graphs  |  Val: {len(val_set)} graphs", flush=True)

    # ── 4. Model ────────────────────────────────────────────────────────
    model     = FIGNNEncoder().to(device)
    count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=50,
    #     T_mult=2,
    #     eta_min=1e-5
    # )
    
    loss_fn = SupConLoss(temperature=0.1)

    # ── 5. Training loop ────────────────────────────────────────────────
    best_sil      = -999
    train_history = []
    val_history   = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # scheduler.step(epoch)
        train_history.append(train_loss)

        # Evaluate every 10 epochs — silhouette on 6k graphs is slow
        if epoch % 10 == 0 or epoch == 1:
            val_z, val_labels = evaluate(model, val_loader, device)
            from sklearn.metrics import silhouette_score
            try:
                sil = silhouette_score(val_z, val_labels)
            except Exception:
                sil = 0.0
            val_history.append(sil)

            print(f"Epoch {epoch:>3}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_silhouette={sil:.3f}  "
                  f"lr={lr:.6f}", flush=True)

            if sil > best_sil:
                best_sil = sil
                torch.save(model.state_dict(), f"{save_dir}/fignn_best.pth")
        else:
            print(f"Epoch {epoch:>3}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}", flush=True)

    # ── 6. Save final weights ───────────────────────────────────────────
    torch.save(model.state_dict(), f"{save_dir}/fignn_final.pth")
    print(f"\nBest val silhouette: {best_sil:.3f}", flush=True)
    print(f"Band importances: {model.get_band_importance()}", flush=True)

    return model, train_history, val_history