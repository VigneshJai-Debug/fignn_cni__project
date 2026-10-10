"""
PLV-threshold sensitivity evaluation for the FIGNN validation pipeline.

This module contains the implementation only. The notebook should keep the
experiment execution and summary cells visible for reproducibility/presentation:

    for threshold in THRESHOLDS:
        graphs_t = make_threshold_graphs(threshold, pre_graph_records)
        result_t = evaluate_threshold(threshold, graphs_t)
        ...

No preprocessing is performed here. `pre_graph_data_current.pkl` is assumed
to already exist and is only read as an input artifact.

The evaluation is frozen:
    - no retraining
    - same LOSO checkpoint for each held-out subject
    - same fold-specific training normalization
    - graph topology is reconstructed at each threshold
    - population trajectory uses training subjects only
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import procrustes
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "data"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "data" /"outputs"

PRE_GRAPH_PATH = (
    DATA_ROOT
    / "processed"
    / "pre_graph_data_reconstructed.pkl"
)

LOSO_ROOT = OUTPUT_ROOT / "loso"
OUTPUT_DIR = OUTPUT_ROOT / "plv_threshold_sensitivity"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
SUBJECTS = list(range(1, 30))

BANDS = ["delta", "theta", "alpha", "beta"]
REST_LABEL = 3
TASK_LABELS = [0, 1, 2]
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model import FIGNNEncoder


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_pre_graph_records(path: Path | str = PRE_GRAPH_PATH) -> list[dict]:
    """Load the precomputed PLV/node-feature records used for reconstruction."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pre-graph data not found: {path}")

    with path.open("rb") as f:
        records = pickle.load(f)

    if not records:
        raise ValueError(f"Pre-graph file is empty: {path}")

    return records


# ---------------------------------------------------------------------------
# Graph reconstruction
# ---------------------------------------------------------------------------

def build_graph_at_threshold(
    connectivity,
    node_features,
    label,
    subject_id,
    session_id,
    threshold,
):
    """Rebuild one graph using mean four-band PLV > threshold for edge selection."""

    n_ch = node_features.shape[0]

    plv_stack = np.stack(
        [connectivity[band] for band in BANDS],
        axis=-1,
    ).astype(np.float32)

    mean_plv = plv_stack.mean(axis=-1)

    upper_triangle = np.triu(
        np.ones((n_ch, n_ch), dtype=bool),
        k=1,
    )

    rows, cols = np.where(
        (mean_plv > threshold) & upper_triangle
    )

    edge_features = plv_stack[rows, cols]

    edge_index = torch.tensor(
        np.stack(
            [
                np.concatenate([rows, cols]),
                np.concatenate([cols, rows]),
            ],
            axis=0,
        ),
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        np.concatenate([edge_features, edge_features], axis=0),
        dtype=torch.float32,
    )

    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
    )

    graph.subject = int(subject_id)
    graph.session = int(session_id)
    graph.plv_threshold = float(threshold)

    return graph


def make_threshold_graphs(threshold, pre_graph_records):
    """Reconstruct every graph from the cached PLV records at one threshold."""

    graphs = []
    edge_counts = []

    for record in pre_graph_records:
        graph = build_graph_at_threshold(
            connectivity=record["connectivity"],
            node_features=record["node_features"],
            label=record["label"],
            subject_id=record["subject_id"],
            session_id=record["session_id"],
            threshold=threshold,
        )

        graph.epoch_idx = int(record["epoch_idx"])
        graph.source_file = record.get("source_file", "")

        graphs.append(graph)
        edge_counts.append(graph.edge_index.shape[1] // 2)

    print(
        f"Threshold {threshold:.1f}: {len(graphs):,} graphs | "
        f"mean undirected edges = {np.mean(edge_counts):.2f}"
    )

    return graphs


# ---------------------------------------------------------------------------
# Frozen fold model + normalization
# ---------------------------------------------------------------------------

def load_fold_normalisation(held_out_subject):
    stats_path = (
        LOSO_ROOT
        / f"subject_{held_out_subject:02d}"
        / "normalisation_stats.pt"
    )

    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float()
    std = stats["std"].float().clamp(min=1e-8)

    return mean, std


def apply_fold_normalisation(graphs, mean, std):
    """Apply the held-out fold's training-only feature normalization."""
    normalized = []

    for graph in graphs:
        graph_copy = copy.copy(graph)
        graph_copy.x = (graph.x.clone() - mean) / std
        normalized.append(graph_copy)

    return normalized


@torch.no_grad()
def encode_graphs(model, graphs):
    """Extract frozen 64-D FIGNN embeddings and labels."""
    loader = DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    embeddings = []
    labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        z = model(batch)

        embeddings.append(z.cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())

    return np.vstack(embeddings), np.concatenate(labels)


def load_fold_model(held_out_subject):
    """Load the already-trained best checkpoint for one LOSO fold."""
    checkpoint = (
        LOSO_ROOT
        / f"subject_{held_out_subject:02d}"
        / "fignn_best.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = FIGNNEncoder().to(DEVICE)
    state_dict = torch.load(checkpoint, map_location=DEVICE)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, checkpoint


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def build_subject_trajectory(z, y):
    """Build the raw 0B -> 1B -> 2B centroid trajectory, matching Stage-1."""
    trajectory = []
    for label in TASK_LABELS:
        mask = y == label
        if mask.sum() == 0:
            raise RuntimeError(f"No samples for label {label}.")
        trajectory.append(z[mask].mean(axis=0))
    trajectory = np.stack(trajectory)
    d02 = np.linalg.norm(trajectory[2] - trajectory[0])
    if d02 < 1e-12:
        raise RuntimeError("0→2 displacement is too small.")
    return trajectory, None, d02


def trajectory_position(point, trajectory):
    """Return continuous projection position t and perpendicular distance."""
    C0, _, C2 = trajectory
    direction = C2 - C0
    denom = np.dot(direction, direction)

    if denom < 1e-12:
        return np.nan, np.nan

    t = np.dot(point - C0, direction) / denom
    projection = C0 + t * direction
    perpendicular_distance = np.linalg.norm(point - projection)

    return float(t), float(perpendicular_distance)


def trajectory_geometry(trajectory):
    """Compute trajectory distances, direction angle, and path efficiency."""
    C0, C1, C2 = trajectory

    d01 = np.linalg.norm(C1 - C0)
    d02 = np.linalg.norm(C2 - C0)
    d12 = np.linalg.norm(C2 - C1)

    if d01 < 1e-12 or d12 < 1e-12:
        angle = np.nan
    else:
        # Angle between the two trajectory directions:
        # C0 -> C1 and C1 -> C2
        v01 = C1 - C0
        v12 = C2 - C1

        cosine = np.dot(v01, v12) / (d01 * d12)
        angle = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )

    efficiency = (
        d02 / (d01 + d12)
        if d01 + d12 > 1e-12
        else np.nan
    )

    return d01, d12, d02, angle, efficiency

# ---------------------------------------------------------------------------
# Main threshold evaluator
# ---------------------------------------------------------------------------

def evaluate_threshold(threshold, all_graphs):
    """Evaluate one threshold across all 29 frozen LOSO folds."""

    print("\n" + "=" * 78)
    print(f"THRESHOLD = {threshold:.1f}")
    print("=" * 78)

    fold_results = []

    for held_out_subject in SUBJECTS:
        print(f"\nSubject {held_out_subject:02d}/29", flush=True)

        train_graphs = [
            g for g in all_graphs
            if int(g.subject) != held_out_subject
        ]
        test_graphs = [
            g for g in all_graphs
            if int(g.subject) == held_out_subject
        ]

        mean, std = load_fold_normalisation(held_out_subject)
        train_graphs_norm = apply_fold_normalisation(train_graphs, mean, std)
        test_graphs_norm = apply_fold_normalisation(test_graphs, mean, std)

        model, checkpoint = load_fold_model(held_out_subject)

        training_subjects = sorted({int(g.subject) for g in train_graphs_norm})
        population_trajectories = []

        for sid in training_subjects:
            sid_graphs = [
                g for g in train_graphs_norm
                if int(g.subject) == sid
            ]

            try:
                z_sid, y_sid = encode_graphs(model, sid_graphs)
                T_sid, _, _ = build_subject_trajectory(z_sid, y_sid)
                population_trajectories.append(T_sid)
            except RuntimeError as exc:
                print(
                    f"  Training subject {sid:02d}: "
                    f"invalid trajectory ({exc})"
                )

        if not population_trajectories:
            print("  No valid population trajectories")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        population_trajectory = np.stack(population_trajectories).mean(axis=0)

        z_subject, y_subject = encode_graphs(model, test_graphs_norm)
        subject_trajectory, _, _ = build_subject_trajectory(z_subject, y_subject)

        d01, d12, d02, angle, efficiency = trajectory_geometry(subject_trajectory)
        d01_d02 = d01 / d02 if d02 > 1e-12 else np.nan
        d12_d02 = d12 / d02 if d02 > 1e-12 else np.nan
        d01_lt_d02 = bool(d01 < d02)

        C0, C1, C2 = subject_trajectory
        relative_position, perpendicular_distance = trajectory_position(C1, subject_trajectory)
        centrality = np.clip(1.0 - 2.0 * abs(relative_position - 0.5), 0.0, 1.0)
        oneback_between_x = bool(0.0 <= relative_position <= 1.0)

        try:
            _, _, disparity = procrustes(subject_trajectory, population_trajectory)
            procrustes_similarity = 1.0 - disparity
        except Exception:
            procrustes_similarity = np.nan

        subject_direction = C2 - C0
        population_direction = population_trajectory[2] - population_trajectory[0]
        denom = np.linalg.norm(subject_direction) * np.linalg.norm(population_direction)
        direction_cosine = (
            np.dot(subject_direction, population_direction) / denom
            if denom > 1e-12
            else np.nan
        )

        fold_results.append({
            "threshold": float(threshold),
            "subject": int(held_out_subject),
            "d01": float(d01),
            "d02": float(d02),
            "d12": float(d12),
            "d01_d02": float(d01_d02),
            "d12_d02": float(d12_d02),
            "angle_deg": float(angle),
            "path_efficiency": float(efficiency),
            "d01_lt_d02": d01_lt_d02,
            "relative_1back_position": float(relative_position),
            "centrality": float(centrality),
            "perpendicular_distance": float(perpendicular_distance),
            "oneback_between_x": oneback_between_x,
            "procrustes_similarity": float(procrustes_similarity),
            "direction_cosine": float(direction_cosine),
            "checkpoint": str(checkpoint),
            "n_training_subjects": len(population_trajectories),
        })

        print(
            f"  d01/d02={d01_d02:.4f} | "
            f"d12/d02={d12_d02:.4f} | "
            f"angle={angle:.2f}° | "
            f"eff={efficiency:.4f} | "
            f"1B position={relative_position:.4f} | "
            f"centrality={centrality:.4f} | "
            f"perp={perpendicular_distance:.4f} | "
            f"between={oneback_between_x}",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(fold_results)


__all__ = [
    "THRESHOLDS",
    "SUBJECTS",
    "OUTPUT_DIR",
    "PRE_GRAPH_PATH",
    "load_pre_graph_records",
    "make_threshold_graphs",
    "evaluate_threshold",
]
