"""
FIGNN architecture sensitivity ablation.
Compares the historical 64/4/32 architecture with the final 128/8/64
architecture under the corrected Stage-1 training protocol.

Protocol:
- COG-BCI, REST removed
- strict 29-fold subject-wise LOSO
- subject-wise 85/15 validation split among the 28 training subjects
- node-feature normalization computed from the training subjects of that split only
- WeightedRandomSampler
- SupConLoss, temperature=0.1
- AdamW, lr=1e-3, weight_decay=1e-4
- CosineAnnealingLR, eta_min=1e-5
- 500 epochs
- best checkpoint selected by lowest validation SupCon loss

This is an architecture-sensitivity experiment; it does not replace the
authoritative final Stage-1 29-fold evaluation.
"""

import os
import sys
import pickle
import subprocess
import random

import torch
import numpy as np

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_metric_learning.losses import SupConLoss

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

DATA_PATH = (
    PROJECT_DIR / "data" / "data" / "processed" / "all_graphs_29sub.pkl"
)

SAVE_ROOT = PROJECT_DIR /"data" /"data" /"outputs" / "ablation_64_4_32"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

ABLATION_SUBJECTS = list(range(1, 30))
N_EPOCHS = 500
BATCH_SIZE = 128
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 42

ABLATION_HIDDEN_DIM = 64
ABLATION_N_HEADS = 4
ABLATION_LATENT_DIM = 32
DROPOUT = 0.3

GPUS = [1, 2, 5, 6]

os.makedirs(SAVE_ROOT, exist_ok=True)
sys.path.insert(0, PROJECT_DIR)

from src.models.model import FIGNNEncoder


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dataset_stats(graph_dataset):
    all_x = torch.cat([g.x for g in graph_dataset], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0).clamp(min=1e-8)
    print(f"  Feature mean: {mean.tolist()}", flush=True)
    print(f"  Feature std:  {std.tolist()}", flush=True)
    return mean, std


def normalise_dataset(graph_dataset, mean, std):
    for g in graph_dataset:
        g.x = (g.x - mean) / std


def make_subjectwise_split(graphs, val_split=VAL_SPLIT, seed=SEED):
    subjects = sorted({int(g.subject) for g in graphs})
    if len(subjects) < 2:
        raise ValueError("Need at least two training subjects for validation.")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_split)))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    train_set = [g for g in graphs if int(g.subject) in train_subjects]
    val_set = [g for g in graphs if int(g.subject) in val_subjects]
    return train_set, val_set, sorted(train_subjects), sorted(val_subjects)


def make_balanced_sampler(dataset):
    labels = [int(g.y.item()) for g in dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = [1.0 / class_counts[l].item() for l in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z = model(batch)
        loss = loss_fn(z, batch.y.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        z = model(batch)
        loss = loss_fn(z, batch.y.squeeze())
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_ablation_training(train_graphs, test_graphs, save_dir, seed=SEED):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print("\nABLATION ARCHITECTURE", flush=True)
    print("hidden_dim = 64", flush=True)
    print("n_heads    = 4", flush=True)
    print("latent_dim = 32", flush=True)

    # Split subjects first; only the resulting training subjects may define normalization.
    train_split, val_set, train_subjects, val_subjects = make_subjectwise_split(
        train_graphs, VAL_SPLIT, seed
    )
    print(f"Training subjects: {train_subjects}", flush=True)
    print(f"Validation subjects: {val_subjects}", flush=True)
    print(f"Train: {len(train_split)} graphs", flush=True)
    print(f"Val:   {len(val_set)} graphs", flush=True)
    print(f"Test:  {len(test_graphs)} graphs", flush=True)

    mean, std = compute_dataset_stats(train_split)
    torch.save(
        {"mean": mean.cpu(), "std": std.cpu()},
        os.path.join(save_dir, "normalisation_stats.pt"),
    )

    normalise_dataset(train_split, mean, std)
    normalise_dataset(val_set, mean, std)
    normalise_dataset(test_graphs, mean, std)

    sampler = make_balanced_sampler(train_split)
    train_loader = DataLoader(
        train_split, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = FIGNNEncoder(
        hidden_dim=ABLATION_HIDDEN_DIM,
        latent_dim=ABLATION_LATENT_DIM,
        n_heads=ABLATION_N_HEADS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )
    loss_fn = SupConLoss(temperature=0.1)

    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            val_loss = evaluate_loss(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch:>3}/{N_EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "fignn_best.pth"))
        else:
            print(
                f"Epoch {epoch:>3}/{N_EPOCHS}  train_loss={train_loss:.4f}",
                flush=True,
            )

    print(f"\nBest val SupCon loss: {best_val_loss:.6f}", flush=True)
    print(f"Band importances: {model.get_band_importance()}", flush=True)

    return {
        "best_val_loss": float(best_val_loss),
        "n_parameters": int(n_params),
        "latent_dim": ABLATION_LATENT_DIM,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
    }


def run_fold(subject, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_seed(SEED)

    print("\n" + "=" * 70, flush=True)
    print(f"ABLATION LOSO FOLD: SUBJECT {subject:02d}", flush=True)
    print(f"GPU: {gpu}", flush=True)
    print("Architecture: 64 hidden / 4 heads / 32 latent", flush=True)
    print("=" * 70, flush=True)

    with open(DATA_PATH, "rb") as f:
        graphs = pickle.load(f)

    graphs = [g for g in graphs if g.y.item() != 3]
    train_graphs = [g for g in graphs if int(g.subject) != subject]
    test_graphs = [g for g in graphs if int(g.subject) == subject]

    train_subjects = sorted({int(g.subject) for g in train_graphs})
    if len(train_subjects) != 28:
        raise ValueError(
            f"Expected 28 training subjects, got {len(train_subjects)} for subject {subject}"
        )

    print(f"Train subjects: {train_subjects}", flush=True)
    print(f"Held-out subject: {subject}", flush=True)
    print(f"Train graphs: {len(train_graphs)}", flush=True)
    print(f"Test graphs: {len(test_graphs)}", flush=True)

    fold_dir = os.path.join(SAVE_ROOT, f"subject_{subject:02d}")
    os.makedirs(fold_dir, exist_ok=True)

    results = run_ablation_training(
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        save_dir=fold_dir,
        seed=SEED,
    )

    with open(os.path.join(fold_dir, "results.txt"), "w") as f:
        f.write(f"subject={subject}\n")
        f.write("hidden_dim=64\n")
        f.write("n_heads=4\n")
        f.write("latent_dim=32\n")
        f.write(f"n_parameters={results['n_parameters']}\n")
        f.write(f"best_val_loss={results['best_val_loss']}\n")
        f.write(f"validation_split_subjects={results['val_subjects']}\n")
        f.write(f"training_split_subjects={results['train_subjects']}\n")

    print(f"\nSUBJECT {subject:02d} COMPLETE", flush=True)


def main():
    remaining = ABLATION_SUBJECTS.copy()

    print("\n" + "=" * 70)
    print("FIGNN ARCHITECTURE ABLATION — FULL 29 FOLDS")
    print("Architecture: 64 hidden / 4 heads / 32 latent")
    print(f"Subjects: {ABLATION_SUBJECTS}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation: subject-wise {VAL_SPLIT:.0%}")
    print(f"Seed: {SEED}")
    print(f"GPUs: {GPUS}")
    print("=" * 70 + "\n")

    while remaining:
        current = []
        for gpu in GPUS:
            if not remaining:
                break
            current.append((remaining.pop(0), gpu))

        processes = []
        for subject, gpu in current:
            print(f"Launching Subject {subject:02d} on GPU {gpu}", flush=True)
            log_path = os.path.join(SAVE_ROOT, f"subject_{subject:02d}.log")
            log_file = open(log_path, "w")
            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            p = subprocess.Popen(
                [sys.executable, __file__, "--worker", str(subject), str(gpu)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=worker_env,
            )
            processes.append((subject, p, log_file))

        for subject, p, log_file in processes:
            code = p.wait()
            log_file.close()
            if code != 0:
                print(f"ERROR: Subject {subject:02d} failed with code {code}", flush=True)
            else:
                print(f"Subject {subject:02d} finished.", flush=True)

    print("\n" + "=" * 70)
    print("ABLATION RUN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_fold(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
