"""
train.py
--------
Training loop for FIGNN Stage 1.

Loss: TripletMarginLoss (from pytorch-metric-learning)
  For each batch, random triplets are mined:
    anchor   = any epoch
    positive = different epoch, same load level (same label)
    negative = epoch from different load level

  Loss pushes: dist(anchor, positive) + margin < dist(anchor, negative)
  After many batches: z-space organises by load level, not by subject.

Why not cross-entropy:
  We don't want to classify yet. We want a meaningful geometry in z-space
  that will generalise to unseen subjects. Cross-entropy would overfit
  to the 3 training subjects. Contrastive loss learns structure.

Training epochs (confusingly same word, different meaning):
  EEG "epochs" = 2-second windows = graph samples (we have ~5400)
  Training "epochs" = full passes through the dataset (we do 50-100)
  One training epoch = model sees all ~5400 graphs once.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
import numpy as np
import os


def train_one_epoch(model, loader, optimizer, loss_fn, miner, device):
    model.train()
    total_loss = 0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        z      = model(batch)                        # (batch_size, latent_dim)
        labels = batch.y.squeeze()                   # (batch_size,)

        # Mine hard triplets within the batch
        hard_pairs = miner(z, labels)
        loss       = loss_fn(z, labels, hard_pairs)

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
    Returns numpy arrays.
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
                 save_dir: str = "outputs/checkpoints",
                 n_epochs: int = 50,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 val_split: float = 0.15):
    """
    Full training run.

    Args:
        graph_dataset : list of PyG Data objects (all subjects, all tasks)
        save_dir      : where to save model weights
        n_epochs      : training epochs (full passes through dataset)
        batch_size    : graphs per batch — reduce to 16 if CUDA OOM
        lr            : learning rate
        val_split     : fraction of data held out for validation
    """
    from src.models.model import FIGNNEncoder, count_parameters

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train / val split
    n_total = len(graph_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    # Shuffle before splitting
    indices   = torch.randperm(n_total).tolist()
    train_set = [graph_dataset[i] for i in indices[:n_train]]
    val_set   = [graph_dataset[i] for i in indices[n_train:]]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_set)} graphs  |  Val: {len(val_set)} graphs")

    # Model
    model     = FIGNNEncoder().to(device)
    count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Loss + miner
    loss_fn = TripletMarginLoss(margin=0.5)
    miner   = TripletMarginMiner(margin=0.5, type_of_triplets="hard")

    # Training loop
    best_val_loss = float("inf")
    train_history = []
    val_history   = []

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, miner, device)
        val_z, val_labels = evaluate(model, val_loader, device)

        # Val loss: rough proxy using sklearn silhouette
        from sklearn.metrics import silhouette_score
        try:
            sil = silhouette_score(val_z, val_labels)
        except Exception:
            sil = 0.0

        scheduler.step()

        train_history.append(train_loss)
        val_history.append(sil)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_silhouette={sil:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

        # Save best model (highest silhouette = better cluster separation)
        if sil > best_val_loss:
            best_val_loss = sil
            torch.save(model.state_dict(), f"{save_dir}/fignn_best.pth")

    # Save final
    torch.save(model.state_dict(), f"{save_dir}/fignn_final.pth")
    print(f"\nBest val silhouette: {best_val_loss:.3f}")
    print(f"Band importances after training: {model.get_band_importance()}")

    return model, train_history, val_history