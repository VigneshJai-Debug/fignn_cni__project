# =============================================================================
# full_train.py
# =============================================================================
# Full-dataset population training
#
# Trains FIGNN on all 29 available subjects.
# This is not an unseen-subject validation experiment.
# =============================================================================

import os
import sys
import json
import pickle
import torch

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------

PROJECT_ROOT = "/home/nvidia/24BCE1793/fignn_cni__project"
DATA_PATH = "/lp-dev/24BCE1793/data/processed/all_graphs_29sub.pkl"

CHECKPOINT_DIR = "/lp-dev/24BCE1793/outputs/checkpoints_full"
PLOT_DIR = "/lp-dev/24BCE1793/outputs/plots_full"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SEED = 42
N_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15

torch.manual_seed(SEED)

# -------------------------------------------------------------------------
# GPU check
# -------------------------------------------------------------------------

print("\nCUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("Visible GPU count:", torch.cuda.device_count(), flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
else:
    raise RuntimeError(
        "CUDA is not available. "
        "Launch with CUDA_VISIBLE_DEVICES=0 or fix the environment."
    )

# -------------------------------------------------------------------------
# Load graphs
# -------------------------------------------------------------------------

print("\nLoading graphs...", flush=True)

with open(DATA_PATH, "rb") as f:
    all_graphs = pickle.load(f)

print(f"Loaded {len(all_graphs)} graphs", flush=True)

# -------------------------------------------------------------------------
# Dataset checks
# -------------------------------------------------------------------------

subjects = sorted(set(int(g.subject) for g in all_graphs))
labels = sorted(set(int(g.y.item()) for g in all_graphs))

print(f"\nSubjects: {subjects}", flush=True)
print(f"Number of subjects: {len(subjects)}", flush=True)
print(f"Labels present: {labels}", flush=True)

for label in labels:
    count = sum(int(g.y.item()) == label for g in all_graphs)
    print(f"Class {label}: {count} graphs", flush=True)

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

from src.training.train import run_training
# from src.evaluation.evaluate import plot_loss_curve

print("\nStarting full-population training...", flush=True)

model, train_hist = run_training(
    train_graphs=all_graphs,
    save_dir=CHECKPOINT_DIR,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    val_split=VAL_SPLIT,
    seed=SEED,
)
# -------------------------------------------------------------------------
# Save metadata
# -------------------------------------------------------------------------

metadata = {
    "experiment": "full_population_training",
    "dataset": "COG-BCI",
    "subjects": subjects,
    "number_of_subjects": len(subjects),
    "number_of_graphs": len(all_graphs),
    "labels": labels,
    "n_epochs": N_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "validation_split": VAL_SPLIT,
    "seed": SEED,
    "note": (
        "Full-dataset population training; "
        "not an unseen-subject validation experiment."
    ),
}

with open(
    os.path.join(CHECKPOINT_DIR, "training_metadata.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)

# -------------------------------------------------------------------------
# Save training curve
# -------------------------------------------------------------------------

# if train_hist:
#     plot_loss_curve(
#         train_hist,
#         None,
#         save_path=os.path.join(
#             PLOT_DIR,
#             "full_population_loss_curve.png",
#         ),
#     )

# -------------------------------------------------------------------------
# Completion
# -------------------------------------------------------------------------

print("\n" + "=" * 70, flush=True)
print("FULL POPULATION MODEL TRAINING COMPLETE", flush=True)
print("=" * 70, flush=True)

print(f"Subjects trained: {len(subjects)}", flush=True)
print(f"Total graphs: {len(all_graphs)}", flush=True)
print(f"\nCheckpoints: {CHECKPOINT_DIR}", flush=True)
print(f"Plots:       {PLOT_DIR}", flush=True)