# =============================================================================
# full_train.py
# =============================================================================
# Final population model training
#
# Train FIGNN on all 29 available subjects.
#
# This is NOT an unseen-subject validation experiment.
# It produces the final frozen population model that can later be applied
# to a genuinely new subject (e.g. Subject 30).
# =============================================================================

import sys
import os
import pickle
import torch

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------

sys.path.insert(0, "/home/nvidia/24BCE1793/fignn_cni__project")
os.chdir("/home/nvidia/24BCE1793/fignn_cni__project")

DATA_PATH = "/lp-dev/24BCE1793/data/processed/all_graphs_29sub.pkl"

CHECKPOINT_DIR = "/lp-dev/24BCE1793/outputs/checkpoints_full"
PLOT_DIR       = "/lp-dev/24BCE1793/outputs/plots_full"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# Load all graphs
# -------------------------------------------------------------------------

print("\nLoading graphs...", flush=True)

with open(DATA_PATH, "rb") as f:
    all_graphs = pickle.load(f)

print(f"Loaded {len(all_graphs)} graphs", flush=True)


# -------------------------------------------------------------------------
# Check dataset
# -------------------------------------------------------------------------

subjects = sorted(set(g.subject for g in all_graphs))

print(f"\nSubjects: {subjects}", flush=True)
print(f"Number of subjects: {len(subjects)}", flush=True)

labels = sorted(set(g.y.item() for g in all_graphs))

print(f"Labels present: {labels}", flush=True)

for label in labels:
    n = sum(g.y.item() == label for g in all_graphs)
    print(f"Class {label}: {n} graphs", flush=True)


# -------------------------------------------------------------------------
# Train final population model
# -------------------------------------------------------------------------

from src.training.train import run_training
from src.evaluation.evaluate import plot_loss_curve


model, train_hist, val_hist, _ = run_training(

    train_graphs=all_graphs,

    # No held-out subject.
    # This is final population training.
    test_graphs=None,

    n_epochs=500,

    batch_size=128,

    lr=1e-3,

    save_dir=CHECKPOINT_DIR
)


# -------------------------------------------------------------------------
# Save training curve
# -------------------------------------------------------------------------

plot_loss_curve(
    train_hist,
    val_hist,
    save_path=os.path.join(PLOT_DIR, "full_population_loss_curve.png")
)


print("\n" + "=" * 70, flush=True)
print("FINAL POPULATION MODEL TRAINING COMPLETE", flush=True)
print("=" * 70, flush=True)

print(f"Subjects trained: {len(subjects)}", flush=True)
print(f"Total graphs: {len(all_graphs)}", flush=True)

print(f"\nCheckpoints saved to:")
print(CHECKPOINT_DIR, flush=True)

print(f"\nPlots saved to:")
print(PLOT_DIR, flush=True)