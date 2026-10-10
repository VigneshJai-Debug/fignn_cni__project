#!/usr/bin/env python3
"""
Controlled 4-fold latent-geometry benchmark for FIGNN.

Purpose
-------
Compare FIGNN against standard graph encoders under the SAME representation-
learning protocol, rather than conventional classification accuracy.

Fixed protocol
--------------
- Dataset: all_graphs_29sub.pkl
- Graphs: authoritative PLV-threshold=0.5 graphs already stored in the pickle
- REST (label 3) removed
- Held-out subjects: 7, 15, 22, 29
- Remaining 28 subjects -> 24 train + 4 validation subjects
- Subject-wise split, seed=42
- Training-only node-feature normalization
- Balanced graph sampler
- SupCon loss, temperature=0.1
- AdamW, lr=1e-3, weight_decay=1e-4
- CosineAnnealingLR, 500 epochs
- Gradient clipping at 1.0
- 64-D L2-normalized embeddings
- Same Stage-1 centroid geometry for every model

Baselines
---------
1. GCN          : standard 2-layer GCN, topology only
2. Vanilla GAT  : standard 2-layer GAT, topology + learned attention,
                   but NO edge attributes / PLV weighting

FIGNN
-----
The existing FIGNN is optionally trained in this script too. By default,
only the two baselines are trained because the controlled 4-fold FIGNN
results already exist from the ablation experiment. Set RUN_FIGNN=True if
you want a fully self-contained 12-run baseline + 4-run FIGNN benchmark.

Important
---------
Do NOT regenerate graphs here. This script deliberately consumes the
authoritative all_graphs_29sub.pkl directly, so the graph topology remains
the existing PLV-mean > 0.5 construction.

Outputs
-------
fignn_cni__project/data/outputs/baseline_benchmark_4fold
    subject_07/
        gcn/
        gat/
    ...
    baseline_geometry_results_4fold.csv
    baseline_geometry_summary_4fold.csv
    benchmark_with_fignn_4fold.csv   (if the existing ablation summary exists)
"""

import os
import sys
import copy
import json
import random
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_mean_pool,
)
from torch.utils.data import WeightedRandomSampler
from pytorch_metric_learning.losses import SupConLoss


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "data" / "processed" / "all_graphs_29sub.pkl"
)

OUTPUT_ROOT = PROJECT_ROOT / "data" / "data" / "outputs" / "baseline_benchmark_4fold"

TEST_SUBJECTS = [7, 15, 22, 29]
BASELINES = ["gcn", "gat"]

# Set True only if you want to retrain FIGNN in this script as well.
# Existing controlled 4-fold FIGNN results can be merged automatically below.
RUN_FIGNN = False

# One worker occupies one physical GPU.
GPUS = [1, 2, 5, 6]

SEED = 42
N_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1
VAL_SPLIT = 0.15
DROPOUT = 0.3
HIDDEN_DIM = 128
LATENT_DIM = 64
GAT_HEADS = 8
NUM_WORKERS = 2

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA / SPLIT
# ============================================================================

def subject_id(graph):
    value = getattr(graph, "subject")
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def label_id(graph):
    return int(graph.y.item())


def load_stage1_graphs():
    """
    Load the authoritative graph pickle and remove REST.

    The pickle already contains the PLV-threshold=0.5 topology. No graph
    rebuilding or thresholding is performed here.
    """
    with open(DATA_PATH, "rb") as f:
        graphs = pickle.load(f)

    graphs = [g for g in graphs if label_id(g) != 3]

    subjects = sorted({subject_id(g) for g in graphs})
    labels = sorted({label_id(g) for g in graphs})

    if subjects != list(range(1, 30)):
        raise RuntimeError(
            f"Expected subjects 1..29, got {subjects}"
        )

    if labels != [0, 1, 2]:
        raise RuntimeError(
            f"Expected Stage-1 labels [0,1,2], got {labels}"
        )

    print(f"Loaded authoritative graphs: {len(graphs)}")
    print(f"Subjects: {subjects}")
    print(f"Labels after REST removal: {labels}")

    return graphs


def make_fold_split(all_graphs, test_subject):
    """
    Exact controlled-ablation split:
      1 held-out test subject
      28 remaining subjects
      4 validation subjects
      24 training subjects
    """
    remaining_subjects = list(
        dict.fromkeys(
            subject_id(g)
            for g in all_graphs
            if subject_id(g) != test_subject
        )
    )

    if len(remaining_subjects) != 28:
        raise RuntimeError(
            f"Expected 28 remaining subjects, got {len(remaining_subjects)}"
        )

    rng = random.Random(SEED)
    shuffled = remaining_subjects.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * VAL_SPLIT)))
    n_val = min(n_val, len(shuffled) - 1)

    val_subjects = set(shuffled[:n_val])
    train_subjects = set(shuffled[n_val:])

    if len(train_subjects) != 24 or len(val_subjects) != 4:
        raise RuntimeError(
            f"Expected 24/4 train/validation subjects, got "
            f"{len(train_subjects)}/{len(val_subjects)}"
        )

    if train_subjects & val_subjects:
        raise RuntimeError("Train/validation subject overlap.")

    if test_subject in train_subjects or test_subject in val_subjects:
        raise RuntimeError("Held-out subject leaked into train/validation.")

    return sorted(train_subjects), sorted(val_subjects)


# ============================================================================
# NORMALIZATION
# ============================================================================

def compute_train_stats(graphs):
    x = torch.cat([g.x for g in graphs], dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-8)
    return mean, std


def normalize_graphs(graphs, mean, std):
    """
    Deep-copy before modifying x so one model/fold never contaminates another.
    """
    graphs = copy.deepcopy(graphs)
    mean = mean.cpu()
    std = std.cpu()

    for g in graphs:
        g.x = (g.x - mean) / std

    return graphs


# ============================================================================
# SAMPLER / LOADERS
# ============================================================================

def make_balanced_sampler(dataset):
    labels = torch.tensor(
        [label_id(g) for g in dataset],
        dtype=torch.long,
    )

    counts = torch.bincount(labels, minlength=3)

    if torch.any(counts == 0):
        raise RuntimeError(
            f"Missing class in training set: counts={counts.tolist()}"
        )

    weights = torch.tensor(
        [1.0 / counts[y].item() for y in labels.tolist()],
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def make_loaders(train_graphs, val_graphs):
    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        sampler=make_balanced_sampler(train_graphs),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================================
# BASELINE ENCODERS
# ============================================================================

class GCNEncoder(nn.Module):
    """
    Standard graph convolution baseline.

    Uses exactly the same node features and fixed graph topology as FIGNN,
    but ignores edge_attr/PLV values. This isolates the contribution of
    attention + frequency-specific edge information.
    """

    def __init__(
        self,
        node_in_dim=4,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index)
        x = self.norm2(x)
        x = F.relu(x)

        z = global_mean_pool(x, data.batch)
        return F.normalize(z, p=2, dim=1)


class VanillaGATEncoder(nn.Module):
    """
    Standard GAT baseline.

    Same 2-layer depth, hidden width, attention-head count and latent
    dimension as FIGNN, but edge_attr is NOT supplied to GATConv.

    Therefore attention is driven by node representations and graph
    topology only; there is no PLV-aware attention or band weighting.
    """

    def __init__(
        self,
        node_in_dim=4,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        heads=GAT_HEADS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=node_in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=latent_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = self.gat1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, data.edge_index)
        x = self.norm2(x)

        z = global_mean_pool(x, data.batch)
        return F.normalize(z, p=2, dim=1)


def build_model(name):
    if name == "gcn":
        return GCNEncoder()

    if name == "gat":
        return VanillaGATEncoder()

    if name == "fignn":
        from src.models.model import FIGNNEncoder
        return FIGNNEncoder(
            node_in_dim=4,
            edge_dim=4,
            hidden_dim=HIDDEN_DIM,
            latent_dim=LATENT_DIM,
            n_heads=GAT_HEADS,
            dropout=DROPOUT,
        )

    raise ValueError(f"Unknown model: {name}")


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        z = model(batch)
        y = batch.y.view(-1).long()

        loss = loss_fn(z, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,)

        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    total = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)

        z = model(batch)
        y = batch.y.view(-1).long()

        loss = loss_fn(z, y)

        total += float(loss.item())
        n += 1

    return total / max(n, 1)


def train_fold(model_name, test_subject, all_graphs, device):
    """
    Train one model for one held-out subject.

    Validation checkpoint is selected by lowest validation SupCon loss.
    The held-out subject is never used for training or validation.
    """
    set_seed(SEED)

    train_subjects, val_subjects = make_fold_split(
        all_graphs,
        test_subject,
    )

    train_raw = [
        g for g in all_graphs
        if subject_id(g) in train_subjects
    ]

    val_raw = [
        g for g in all_graphs
        if subject_id(g) in val_subjects
    ]

    test_raw = [
        g for g in all_graphs
        if subject_id(g) == test_subject
    ]

    mean, std = compute_train_stats(train_raw)

    train_graphs = normalize_graphs(train_raw, mean, std)
    val_graphs = normalize_graphs(val_raw, mean, std)
    test_graphs = normalize_graphs(test_raw, mean, std)

    train_loader, val_loader = make_loaders(
        train_graphs,
        val_graphs,
    )

    model = build_model(model_name).to(device)

    print(
        f"[{model_name} | test={test_subject:02d}] "
        f"parameters="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=N_EPOCHS,
        eta_min=1e-5,
    )

    loss_fn = SupConLoss(temperature=TEMPERATURE)

    best_val = float("inf")
    best_epoch = None
    history = []

    fold_dir = OUTPUT_ROOT / f"subject_{test_subject:02d}" / model_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = fold_dir / "best.pt"

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss = validate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "test_subject": test_subject,
                    "train_subjects": train_subjects,
                    "validation_subjects": val_subjects,
                    "feature_mean": mean.cpu(),
                    "feature_std": std.cpu(),
                    "best_validation_loss": best_val,
                    "epoch": best_epoch,
                    "seed": SEED,
                    "config": {
                        "n_epochs": N_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "lr": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "temperature": TEMPERATURE,
                        "hidden_dim": HIDDEN_DIM,
                        "latent_dim": LATENT_DIM,
                        "gat_heads": GAT_HEADS,
                        "dropout": DROPOUT,
                    },
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % 50 == 0 or epoch == N_EPOCHS:
            print(
                f"[{model_name} | test={test_subject:02d}] "
                f"epoch {epoch:03d}/{N_EPOCHS} "
                f"train={train_loss:.5f} "
                f"val={val_loss:.5f}"
            )

    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"[{model_name} | test={test_subject:02d}] "
        f"best epoch={best_epoch}, val={best_val:.6f}"
    )

    return checkpoint_path


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_embeddings(model, graphs, device):
    loader = DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    z_all = []
    y_all = []
    s_all = []

    model.eval()

    for batch in loader:
        batch = batch.to(device)

        z = model(batch)

        z_all.append(z.cpu())
        y_all.append(batch.y.view(-1).cpu())
        s_all.extend(
            batch.subject.view(-1).cpu().tolist()
        )

    return (
        torch.cat(z_all, dim=0).numpy(),
        torch.cat(y_all, dim=0).numpy(),
        np.asarray(s_all, dtype=int),
    )


def load_best_model(model_name, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ============================================================================
# GEOMETRY
# Exact formulas used by the existing controlled 4-fold evaluator
# ============================================================================

def build_trajectory(embeddings, labels):
    centroids = []

    for class_id in [0, 1, 2]:
        mask = labels == class_id

        if not np.any(mask):
            return None

        centroids.append(
            embeddings[mask].mean(axis=0)
        )

    return np.stack(centroids, axis=0)


def trajectory_geometry(T):
    C0, C1, C2 = T

    d01 = np.linalg.norm(C1 - C0)
    d12 = np.linalg.norm(C2 - C1)
    d02 = np.linalg.norm(C2 - C0)

    # Angle at 1-back centroid.
    v1 = C0 - C1
    v2 = C2 - C1

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom < 1e-12:
        angle_deg = np.nan
    else:
        cosine = np.dot(v1, v2) / denom
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine))

    path_efficiency = (
        d02 / (d01 + d12)
        if (d01 + d12) > 1e-12
        else np.nan
    )

    return {
        "d01": d01,
        "d12": d12,
        "d02": d02,
        "d01_d02": (
            d01 / d02 if d02 > 1e-12 else np.nan
        ),
        "d12_d02": (
            d12 / d02 if d02 > 1e-12 else np.nan
        ),
        "angle_deg": angle_deg,
        "path_efficiency": path_efficiency,
    }


def relative_position_metrics(T):
    C0, C1, C2 = T

    direction = C2 - C0
    denom = np.dot(direction, direction)

    if denom < 1e-12:
        return np.nan, np.nan, np.nan, False

    t = np.dot(C1 - C0, direction) / denom

    projection = C0 + t * direction

    perpendicular_distance = np.linalg.norm(
        C1 - projection
    )

    centrality = 1.0 - 2.0 * abs(t - 0.5)
    centrality = float(np.clip(centrality, 0.0, 1.0))

    between = bool(0.0 <= t <= 1.0)

    return (
        float(t),
        float(perpendicular_distance),
        centrality,
        between,
    )


def procrustes_similarity(A, B):
    """
    Same implementation used by the existing controlled ablation evaluator.
    """
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)

    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    if norm_A < 1e-12 or norm_B < 1e-12:
        return np.nan

    A = A / norm_A
    B = B / norm_B

    M = A.T @ B

    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt

    aligned_A = A @ R

    error = np.linalg.norm(
        aligned_A - B
    )

    return float(
        np.clip(
            1.0 - error / np.sqrt(2.0),
            0.0,
            1.0,
        )
    )


def direction_cosine(subject_T, population_T):
    a = subject_T[2] - subject_T[0]
    b = population_T[2] - population_T[0]

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom < 1e-12:
        return np.nan

    return float(
        np.dot(a, b) / denom
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_fold(model_name, test_subject, all_graphs, device):
    fold_dir = OUTPUT_ROOT / f"subject_{test_subject:02d}" / model_name
    checkpoint_path = fold_dir / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    model, checkpoint = load_best_model(
        model_name,
        checkpoint_path,
        device,
    )

    train_subjects = set(
        int(x) for x in checkpoint["train_subjects"]
    )
    val_subjects = set(
        int(x) for x in checkpoint["validation_subjects"]
    )

    if len(train_subjects) != 24:
        raise RuntimeError("Checkpoint does not contain 24 train subjects.")

    if len(val_subjects) != 4:
        raise RuntimeError("Checkpoint does not contain 4 validation subjects.")

    if test_subject in train_subjects or test_subject in val_subjects:
        raise RuntimeError("Held-out subject leakage detected.")

    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]

    train_raw = [
        g for g in all_graphs
        if subject_id(g) in train_subjects
    ]

    test_raw = [
        g for g in all_graphs
        if subject_id(g) == test_subject
    ]

    train_graphs = normalize_graphs(
        train_raw,
        mean,
        std,
    )

    test_graphs = normalize_graphs(
        test_raw,
        mean,
        std,
    )

    # ---------------------------------------------------------
    # TRAINING-SUBJECT POPULATION TRAJECTORY
    # ---------------------------------------------------------

    train_z, train_y, train_sid = extract_embeddings(
        model,
        train_graphs,
        device,
    )

    training_trajectories = []

    for sid in sorted(train_subjects):
        mask = train_sid == sid

        T = build_trajectory(
            train_z[mask],
            train_y[mask],
        )

        if T is not None:
            training_trajectories.append(T)

    if len(training_trajectories) != 24:
        raise RuntimeError(
            f"Expected 24 training trajectories, got "
            f"{len(training_trajectories)}"
        )

    population_T = np.stack(
        training_trajectories,
        axis=0,
    ).mean(axis=0)

    # ---------------------------------------------------------
    # HELD-OUT SUBJECT
    # ---------------------------------------------------------

    test_z, test_y, _ = extract_embeddings(
        model,
        test_graphs,
        device,
    )

    subject_T = build_trajectory(
        test_z,
        test_y,
    )

    if subject_T is None:
        raise RuntimeError(
            f"Incomplete 0B/1B/2B trajectory for subject {test_subject}"
        )

    # ---------------------------------------------------------
    # GEOMETRY
    # ---------------------------------------------------------

    geom = trajectory_geometry(subject_T)

    (
        relative_1b,
        perpendicular_distance,
        centrality,
        between,
    ) = relative_position_metrics(subject_T)

    proc = procrustes_similarity(
        subject_T,
        population_T,
    )

    dir_cos = direction_cosine(
        subject_T,
        population_T,
    )

    return {
        "test_subject": test_subject,
        "model": model_name,
        "train_subject_count": 24,
        "validation_subject_count": 4,
        "population_subject_count": 24,
        "best_epoch": checkpoint.get("epoch", np.nan),
        "best_validation_loss": checkpoint.get(
            "best_validation_loss",
            np.nan,
        ),
        "d01": geom["d01"],
        "d12": geom["d12"],
        "d02": geom["d02"],
        "d01_d02": geom["d01_d02"],
        "d12_d02": geom["d12_d02"],
        "angle_deg": geom["angle_deg"],
        "path_efficiency": geom["path_efficiency"],
        "relative_1back_position": relative_1b,
        "perpendicular_distance": perpendicular_distance,
        "centrality": centrality,
        "between": between,
        "procrustes_similarity": proc,
        "direction_cosine": dir_cos,
    }


def summarize(results_df):
    rows = []

    for model_name in sorted(results_df["model"].unique()):
        df = results_df[
            results_df["model"] == model_name
        ].copy()

        if len(df) != 4:
            raise RuntimeError(
                f"{model_name}: expected 4 folds, got {len(df)}"
            )

        rows.append(
            {
                "model": model_name,
                "n_folds": len(df),

                "d01_d02_mean":
                    df["d01_d02"].mean(),
                "d01_d02_sd":
                    df["d01_d02"].std(ddof=1),

                "d12_d02_mean":
                    df["d12_d02"].mean(),
                "d12_d02_sd":
                    df["d12_d02"].std(ddof=1),

                "angle_deg_mean":
                    df["angle_deg"].mean(),
                "angle_deg_sd":
                    df["angle_deg"].std(ddof=1),

                "path_efficiency_mean":
                    df["path_efficiency"].mean(),
                "path_efficiency_sd":
                    df["path_efficiency"].std(ddof=1),

                "relative_1back_position_mean":
                    df["relative_1back_position"].mean(),
                "relative_1back_position_sd":
                    df["relative_1back_position"].std(ddof=1),

                "perpendicular_distance_mean":
                    df["perpendicular_distance"].mean(),
                "perpendicular_distance_sd":
                    df["perpendicular_distance"].std(ddof=1),

                "centrality_mean":
                    df["centrality"].mean(),
                "centrality_sd":
                    df["centrality"].std(ddof=1),

                "between_count":
                    int(df["between"].sum()),
                "between_rate":
                    df["between"].mean(),

                "procrustes_mean":
                    df["procrustes_similarity"].mean(),
                "procrustes_sd":
                    df["procrustes_similarity"].std(ddof=1),

                "direction_cosine_mean":
                    df["direction_cosine"].mean(),
                "direction_cosine_sd":
                    df["direction_cosine"].std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# OPTIONAL MERGE WITH EXISTING FIGNN 4-FOLD ABLATION RESULTS
# ============================================================================

def merge_existing_fignn(summary_df):
    """
    If the existing controlled ablation summary is available, pull its
    'full' row into a comparison table. No FIGNN result is recomputed here.
    """
    source = (
        PROJECT_ROOT
        / "outputs"
        / "ablation_logs"
        / "ablation_geometry_summary_4fold.csv"
    )

    if not source.exists():
        return summary_df

    fignn = pd.read_csv(source)

    if "variant" not in fignn.columns:
        return summary_df

    fignn = fignn[fignn["variant"] == "full"].copy()

    if len(fignn) != 1:
        return summary_df

    row = fignn.iloc[0]

    fignn_row = {
        "model": "fignn",
        "n_folds": int(row["n_folds"]),
        "d01_d02_mean": row["d01_d02_mean"],
        "d01_d02_sd": row["d01_d02_sd"],
        "d12_d02_mean": row["d12_d02_mean"],
        "d12_d02_sd": row["d12_d02_sd"],
        "angle_deg_mean": row["angle_deg_mean"],
        "angle_deg_sd": row["angle_deg_sd"],
        "path_efficiency_mean": row["path_efficiency_mean"],
        "path_efficiency_sd": row["path_efficiency_sd"],
        "relative_1back_position_mean":
            row["relative_1back_position_mean"],
        "relative_1back_position_sd":
            row["relative_1back_position_sd"],
        "perpendicular_distance_mean":
            row["perpendicular_distance_mean"],
        "perpendicular_distance_sd":
            row["perpendicular_distance_sd"],
        "centrality_mean":
            row["centrality_mean"],
        "centrality_sd":
            row["centrality_sd"],
        "between_count":
            row["between_count"],
        "between_rate":
            row["between_rate"],
        "procrustes_mean":
            row["procrustes_mean"],
        "procrustes_sd":
            row["procrustes_sd"],
        "direction_cosine_mean":
            row["direction_cosine_mean"],
        "direction_cosine_sd":
            row["direction_cosine_sd"],
    }

    return pd.concat(
        [pd.DataFrame([fignn_row]), summary_df],
        ignore_index=True,
    )


# ============================================================================
# SINGLE-FOLD WORKER
# ============================================================================

def worker(model_name, test_subject, gpu):
    # CUDA_VISIBLE_DEVICES was set before the worker Python process started.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this benchmark.")

    set_seed(SEED)

    print("=" * 80)
    print(
        f"WORKER | model={model_name} | "
        f"test_subject={test_subject:02d} | "
        f"physical_gpu={gpu}"
    )
    print("=" * 80)

    all_graphs = load_stage1_graphs()

    checkpoint_path = (
        OUTPUT_ROOT
        / f"subject_{test_subject:02d}"
        / model_name
        / "best.pt"
    )

    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
    else:
        train_fold(
            model_name,
            test_subject,
            all_graphs,
            device,
        )

    result = evaluate_fold(
        model_name,
        test_subject,
        all_graphs,
        device,
    )

    result_path = (
        OUTPUT_ROOT
        / f"subject_{test_subject:02d}"
        / model_name
        / "geometry_result.json"
    )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2,default=lambda x: x.item() if hasattr(x,"item") else x)

    print(
        f"COMPLETE | {model_name} | "
        f"subject={test_subject:02d} | "
        f"between={result['between']} | "
        f"d01/d02={result['d01_d02']:.4f} | "
        f"path={result['path_efficiency']:.4f} | "
        f"procrustes={result['procrustes_similarity']:.4f}"
    )


# ============================================================================
# LAUNCHER
# ============================================================================

def launch_all():
    """
    Run one subject/model job per GPU, four jobs at a time.
    """
    jobs = []

    model_names = list(BASELINES)

    if RUN_FIGNN:
        model_names.append("fignn")

    for model_name in model_names:
        for subject in TEST_SUBJECTS:
            jobs.append((model_name, subject))

    print("=" * 80)
    print("CONTROLLED 4-FOLD LATENT-GEOMETRY BENCHMARK")
    print("=" * 80)
    print(f"Models        : {model_names}")
    print(f"Test subjects : {TEST_SUBJECTS}")
    print(f"Jobs          : {len(jobs)}")
    print(f"GPUs          : {GPUS}")
    print("=" * 80)

    while jobs:
        current = []

        for gpu in GPUS:
            if not jobs:
                break
            current.append((jobs.pop(0), gpu))

        processes = []

        for (model_name, subject), gpu in current:
            log_path = (
                OUTPUT_ROOT
                / f"subject_{subject:02d}"
                / f"{model_name}.log"
            )

            log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            log_file = open(log_path, "w")

            cmd = [
                sys.executable,
                __file__,
                "--worker",
                model_name,
                str(subject),
                str(gpu),
            ]

            print(
                f"Launching {model_name:>5s} | "
                f"subject {subject:02d} | GPU {gpu}"
            )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            p = subprocess.Popen(cmd,stdout=log_file,stderr=subprocess.STDOUT,env=env)

            processes.append(
                (model_name, subject, p, log_file)
            )

        for model_name, subject, p, log_file in processes:
            code = p.wait()
            log_file.close()

            if code != 0:
                print(
                    f"ERROR: {model_name} subject {subject:02d} "
                    f"failed with code {code}"
                )
            else:
                print(
                    f"Finished: {model_name} subject {subject:02d}"
                )


# ============================================================================
# COLLECT RESULTS
# ============================================================================

def collect_results():
    model_names = list(BASELINES)

    if RUN_FIGNN:
        model_names.append("fignn")

    rows = []

    for model_name in model_names:
        for subject in TEST_SUBJECTS:
            path = (
                OUTPUT_ROOT
                / f"subject_{subject:02d}"
                / model_name
                / "geometry_result.json"
            )

            if not path.exists():
                print(f"[missing] {path}")
                continue

            with open(path) as f:
                rows.append(json.load(f))

    if not rows:
        raise RuntimeError("No geometry results found.")

    df = pd.DataFrame(rows)

    expected = len(model_names) * len(TEST_SUBJECTS)

    if len(df) != expected:
        raise RuntimeError(
            f"Expected {expected} results, found {len(df)}."
        )

    raw_path = OUTPUT_ROOT / "baseline_geometry_results_4fold.csv"
    summary_path = OUTPUT_ROOT / "baseline_geometry_summary_4fold.csv"

    df = df.sort_values(["model", "test_subject"])
    summary = summarize(df)

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    comparison = merge_existing_fignn(summary)

    comparison_path = OUTPUT_ROOT / "benchmark_with_fignn_4fold.csv"
    comparison.to_csv(comparison_path, index=False)

    print()
    print("=" * 120)
    print("RAW 4-FOLD GEOMETRY RESULTS")
    print("=" * 120)

    cols = [
        "test_subject",
        "model",
        "d01_d02",
        "d12_d02",
        "angle_deg",
        "path_efficiency",
        "relative_1back_position",
        "perpendicular_distance",
        "centrality",
        "between",
        "procrustes_similarity",
        "direction_cosine",
    ]

    print(df[cols].round(4).to_string(index=False))

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(comparison.round(4).to_string(index=False))

    print()
    print(f"Raw results      : {raw_path}")
    print(f"Summary          : {summary_path}")
    print(f"FIGNN comparison : {comparison_path}")
    print("=" * 120)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    if "--worker" in sys.argv:
        if len(sys.argv) != 5:
            raise SystemExit(
                "Usage: python baseline_benchmark_4fold.py "
                "--worker <gcn|gat|fignn> <subject> <gpu>"
            )

        model_name = sys.argv[2]
        test_subject = int(sys.argv[3])
        gpu = int(sys.argv[4])

        allowed = set(BASELINES)
        if RUN_FIGNN:
            allowed.add("fignn")

        if model_name not in allowed:
            raise SystemExit(
                f"Unknown model '{model_name}'. Allowed: {sorted(allowed)}"
            )

        if test_subject not in TEST_SUBJECTS:
            raise SystemExit(
                f"Invalid test subject {test_subject}. Use {TEST_SUBJECTS}"
            )

        worker(model_name, test_subject, gpu)

    elif "--collect" in sys.argv:
        collect_results()

    else:
        launch_all()
        collect_results()
