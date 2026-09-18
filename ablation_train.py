# =============================================================================
# ablation_train.py
#
# Subject-29 held-out ablation study for FIGNN Stage 1.
#
# Protocol aligned with the main LOSO training pipeline:
#   - REST (label 3) removed
#   - Subject 29 completely held out
#   - subject-wise 15% validation split
#   - training-only normalization
#   - balanced graph sampling
#   - AdamW
#   - cosine LR schedule
#   - gradient clipping
#
# Checkpoints:
#   <variant>_best.pt   -> lowest TRAINING loss
#   <variant>_model.pt  -> final epoch
#
# Variants:
#   full
#   no_plv
#   no_band_weight
#   no_gat
#   no_supcon
#   band_power
# =============================================================================

import os
import sys
import copy
import json
import random
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch.utils.data import WeightedRandomSampler

from pytorch_metric_learning.losses import SupConLoss


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = "/home/nvidia/24BCE1793/fignn_cni__project"

DATA_PATH = (
    "/lp-dev/24BCE1793/data/processed/"
    "all_graphs_29sub.pkl"
)

OUTPUT_DIR = (
    "/lp-dev/24BCE1793/outputs/"
    "ablation_logs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

os.chdir(PROJECT_ROOT)


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

N_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TEMPERATURE = 0.1

TEST_SUBJECT = 29
VAL_SPLIT = 0.15

VARIANT = os.environ.get(
    "ABLATION_VARIANT",
    "full"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# VALIDATE VARIANT
# =============================================================================

VALID_VARIANTS = {
    "full",
    "no_plv",
    "no_band_weight",
    "no_gat",
    "no_supcon",
    "band_power",
}

if VARIANT not in VALID_VARIANTS:
    raise ValueError(
        f"Invalid variant '{VARIANT}'. "
        f"Choose from {sorted(VALID_VARIANTS)}"
    )


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =============================================================================
# DEVICE
# =============================================================================

print(
    f"Variant: {VARIANT}",
    flush=True
)

print(
    f"CUDA available: {torch.cuda.is_available()}",
    flush=True
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

print(
    f"GPU: {torch.cuda.get_device_name(0)}",
    flush=True
)


# =============================================================================
# LOAD DATA
# =============================================================================

print(
    "\nLoading graphs...",
    flush=True
)

with open(
    DATA_PATH,
    "rb"
) as f:

    all_graphs = pickle.load(f)

print(
    f"Loaded {len(all_graphs)} graphs",
    flush=True
)


# =============================================================================
# STAGE 1:
# REMOVE REST
# =============================================================================

all_graphs = [
    g
    for g in all_graphs
    if int(g.y.item()) != 3
]

print(
    f"After removing REST: "
    f"{len(all_graphs)} graphs",
    flush=True
)


# =============================================================================
# SUBJECT / LABEL CHECK
# =============================================================================

subjects = sorted(
    set(
        int(g.subject)
        for g in all_graphs
    )
)

labels = sorted(
    set(
        int(g.y.item())
        for g in all_graphs
    )
)

N_CLASSES = len(labels)

print(
    "Subjects:",
    subjects,
    flush=True
)

print(
    "Labels:",
    labels,
    flush=True
)

if labels != [0, 1, 2]:
    raise RuntimeError(
        f"Expected labels [0, 1, 2], "
        f"got {labels}"
    )


# =============================================================================
# SUBJECT 29 TEST HOLDOUT
# =============================================================================

test_graphs = [
    g
    for g in all_graphs
    if int(g.subject) == TEST_SUBJECT
]

remaining_graphs = [
    g
    for g in all_graphs
    if int(g.subject) != TEST_SUBJECT
]

remaining_subjects = sorted(
    set(
        int(g.subject)
        for g in remaining_graphs
    )
)


# =============================================================================
# SAME SUBJECT-WISE SPLIT AS train.py
# =============================================================================

rng = random.Random(SEED)

shuffled_subjects = remaining_subjects.copy()

rng.shuffle(
    shuffled_subjects
)

n_val_subjects = max(
    1,
    int(
        round(
            len(shuffled_subjects)
            * VAL_SPLIT
        )
    )
)

n_val_subjects = min(
    n_val_subjects,
    len(shuffled_subjects) - 1
)

val_subjects = set(
    shuffled_subjects[
        :n_val_subjects
    ]
)

train_subjects = set(
    shuffled_subjects[
        n_val_subjects:
    ]
)

train_graphs = [
    g
    for g in remaining_graphs
    if int(g.subject) in train_subjects
]

val_graphs = [
    g
    for g in remaining_graphs
    if int(g.subject) in val_subjects
]


print(
    "\n" + "=" * 70,
    flush=True
)

print(
    "SUBJECT SPLIT",
    flush=True
)

print(
    "=" * 70,
    flush=True
)

print(
    "Train subjects:",
    sorted(train_subjects),
    flush=True
)

print(
    "Validation subjects:",
    sorted(val_subjects),
    flush=True
)

print(
    "Test subject:",
    TEST_SUBJECT,
    flush=True
)

print(
    f"Train graphs: {len(train_graphs)}",
    flush=True
)

print(
    f"Validation graphs: {len(val_graphs)}",
    flush=True
)

print(
    f"Test graphs: {len(test_graphs)}",
    flush=True
)


# =============================================================================
# TRAIN-ONLY NORMALIZATION
# =============================================================================

print(
    "\nComputing training normalization...",
    flush=True
)

train_x = torch.cat(
    [
        g.x
        for g in train_graphs
    ],
    dim=0
)

feature_mean = train_x.mean(
    dim=0
)

feature_std = train_x.std(
    dim=0
).clamp(
    min=1e-8
)

print(
    "Mean:",
    feature_mean.tolist(),
    flush=True
)

print(
    "Std:",
    feature_std.tolist(),
    flush=True
)


def normalize_graphs(graphs):

    graphs = copy.deepcopy(
        graphs
    )

    for graph in graphs:

        graph.x = (
            graph.x
            - feature_mean
        ) / feature_std

    return graphs


train_graphs = normalize_graphs(
    train_graphs
)

val_graphs = normalize_graphs(
    val_graphs
)

test_graphs = normalize_graphs(
    test_graphs
)


# =============================================================================
# ABLATION TRANSFORMATION
# =============================================================================

def apply_ablation(
    graphs,
    variant
):

    graphs = copy.deepcopy(
        graphs
    )

    # ---------------------------------------------------------
    # NO PLV
    #
    # Remove edge attributes while retaining graph topology.
    # ---------------------------------------------------------

    if variant == "no_plv":

        for graph in graphs:

            graph.edge_attr = (
                torch.zeros_like(
                    graph.edge_attr
                )
            )

    return graphs


# =============================================================================
# BALANCED SAMPLER
# =============================================================================

def make_balanced_sampler(
    dataset
):

    labels_local = torch.tensor(
        [
            int(g.y.item())
            for g in dataset
        ],
        dtype=torch.long
    )

    class_counts = torch.bincount(
        labels_local,
        minlength=N_CLASSES
    )

    if torch.any(
        class_counts == 0
    ):
        raise RuntimeError(
            "Missing class in training set."
        )

    weights = torch.tensor(
        [
            1.0
            / class_counts[label].item()
            for label in labels_local
        ],
        dtype=torch.double
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )


# =============================================================================
# LOADERS
# =============================================================================

def make_loaders(
    train_set,
    val_set
):

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=make_balanced_sampler(
            train_set
        ),
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

    return (
        train_loader,
        val_loader
    )


# =============================================================================
# NO-GAT ENCODER
# =============================================================================

class NoGATEncoder(
    nn.Module
):

    def __init__(
        self,
        node_in_dim=4,
        hidden_dim=128,
        latent_dim=64
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                node_in_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )

    def forward(
        self,
        data
    ):

        x = self.network(
            data.x
        )

        z = global_mean_pool(
            x,
            data.batch
        )

        return F.normalize(
            z,
            p=2,
            dim=1
        )


# =============================================================================
# CLASSIFIER
# =============================================================================

class EmbeddingClassifier(
    nn.Module
):

    def __init__(
        self,
        encoder,
        latent_dim=64,
        n_classes=3
    ):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Linear(
            latent_dim,
            n_classes
        )

    def forward(
        self,
        data
    ):

        z = self.encoder(
            data
        )

        return self.classifier(
            z
        )


# =============================================================================
# BAND-POWER BASELINE
# =============================================================================

class BandPowerClassifier(
    nn.Module
):

    def __init__(
        self,
        n_classes=3
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                4,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                n_classes
            )
        )

    def forward(
        self,
        data
    ):

        x = global_mean_pool(
            data.x,
            data.batch
        )

        return self.classifier(
            x
        )


# =============================================================================
# MODEL CREATION
# =============================================================================

def make_encoder(
    variant
):

    from src.models.model import FIGNNEncoder

    if variant == "no_gat":

        return NoGATEncoder(
            node_in_dim=4,
            hidden_dim=128,
            latent_dim=64
        )

    model = FIGNNEncoder()

    # Equal band weighting instead of learned weighting.
    if variant == "no_band_weight":

        with torch.no_grad():

            model.band_weight.raw_weights.fill_(
                1.0
            )

        model.band_weight.raw_weights.requires_grad_(
            False
        )

    return model


# =============================================================================
# SUPCON TRAINING
# =============================================================================

def train_supcon(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn
):

    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:

        batch = batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        z = model(
            batch
        )

        labels_batch = (
            batch.y
            .view(-1)
            .long()
        )

        loss = loss_fn(
            z,
            labels_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        n_batches += 1

    scheduler.step()

    return (
        total_loss
        / max(n_batches, 1)
    )


# =============================================================================
# CROSS-ENTROPY TRAINING
# =============================================================================

def train_ce(
    model,
    loader,
    optimizer,
    scheduler
):

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:

        batch = batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            batch
        )

        labels_batch = (
            batch.y
            .view(-1)
            .long()
        )

        loss = criterion(
            logits,
            labels_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        n_batches += 1

    scheduler.step()

    return (
        total_loss
        / max(n_batches, 1)
    )


# =============================================================================
# VALIDATION ACCURACY
# MONITORING ONLY
# =============================================================================

@torch.no_grad()
def classifier_accuracy(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    for batch in loader:

        batch = batch.to(
            DEVICE
        )

        logits = model(
            batch
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        labels_batch = (
            batch.y
            .view(-1)
            .long()
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.numel()

    return (
        correct / total
        if total > 0
        else 0.0
    )


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================

def checkpoint_payload(
    model,
    epoch,
    train_loss
):

    return {
        "variant": VARIANT,

        "epoch": epoch,

        "train_loss": train_loss,

        "test_subject":
            TEST_SUBJECT,

        "train_subjects":
            sorted(train_subjects),

        "validation_subjects":
            sorted(val_subjects),

        "labels":
            labels,

        "number_of_classes":
            N_CLASSES,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "temperature":
            TEMPERATURE,

        "feature_mean":
            feature_mean.cpu(),

        "feature_std":
            feature_std.cpu(),

        "model_state_dict":
            model.state_dict(),
    }


def save_checkpoint(
    model,
    epoch,
    train_loss,
    filename
):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    torch.save(
        checkpoint_payload(
            model,
            epoch,
            train_loss
        ),
        path
    )

    return path


# =============================================================================
# TRAIN ONE VARIANT
# =============================================================================

def run_variant(
    variant
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        f"STARTING: {variant}",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    # ---------------------------------------------------------
    # Apply graph ablation
    # ---------------------------------------------------------

    train_set = apply_ablation(
        train_graphs,
        variant
    )

    val_set = apply_ablation(
        val_graphs,
        variant
    )

    (
        train_loader,
        val_loader
    ) = make_loaders(
        train_set,
        val_set
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    if variant == "band_power":

        model = BandPowerClassifier(
            n_classes=N_CLASSES
        ).to(DEVICE)

        training_mode = "ce"

    elif variant == "no_supcon":

        encoder = make_encoder(
            variant
        )

        model = EmbeddingClassifier(
            encoder=encoder,
            latent_dim=64,
            n_classes=N_CLASSES
        ).to(DEVICE)

        training_mode = "ce"

    else:

        model = make_encoder(
            variant
        ).to(DEVICE)

        training_mode = "supcon"

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=N_EPOCHS,
            eta_min=1e-5
        )
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    if training_mode == "supcon":

        loss_fn = SupConLoss(
            temperature=TEMPERATURE
        )

    # ---------------------------------------------------------
    # Best = LOWEST TRAINING LOSS
    # ---------------------------------------------------------

    best_train_loss = float(
        "inf"
    )

    best_epoch = None

    best_path = os.path.join(
        OUTPUT_DIR,
        f"{variant}_best.pt"
    )

    history = []

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    for epoch in range(
        1,
        N_EPOCHS + 1
    ):

        if training_mode == "supcon":

            train_loss = train_supcon(
                model,
                train_loader,
                optimizer,
                scheduler,
                loss_fn
            )

        else:

            train_loss = train_ce(
                model,
                train_loader,
                optimizer,
                scheduler
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss
            }
        )

        # -----------------------------------------------------
        # Best checkpoint
        # -----------------------------------------------------

        if train_loss < best_train_loss:

            best_train_loss = train_loss

            best_epoch = epoch

            save_checkpoint(
                model,
                epoch,
                train_loss,
                f"{variant}_best.pt"
            )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == best_epoch
        ):

            print(
                f"Epoch {epoch:3d}/{N_EPOCHS} "
                f"| train_loss={train_loss:.6f} "
                f"| best={best_train_loss:.6f} "
                f"| best_epoch={best_epoch}",
                flush=True
            )

    # ---------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------

    final_path = save_checkpoint(
        model,
        N_EPOCHS,
        history[-1]["train_loss"],
        f"{variant}_model.pt"
    )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        f"{variant}_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "variant": variant,
                "best_epoch": best_epoch,
                "best_train_loss": best_train_loss,
                "final_train_loss":
                    history[-1]["train_loss"],
                "history": history,
            },
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Validation diagnostic
    # ---------------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    if training_mode == "ce":

        val_acc = classifier_accuracy(
            model,
            val_loader
        )

    else:

        val_acc = None

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70,
        flush=True
    )

    print(
        f"{variant} COMPLETE",
        flush=True
    )

    print(
        f"Best epoch: {best_epoch}",
        flush=True
    )

    print(
        f"Best train loss: "
        f"{best_train_loss:.6f}",
        flush=True
    )

    print(
        f"Final train loss: "
        f"{history[-1]['train_loss']:.6f}",
        flush=True
    )

    if val_acc is not None:

        print(
            f"Best-checkpoint validation accuracy: "
            f"{val_acc:.4f}",
            flush=True
        )

    print(
        f"Best checkpoint: {best_path}",
        flush=True
    )

    print(
        f"Final checkpoint: {final_path}",
        flush=True
    )

    print(
        "-" * 70,
        flush=True
    )

    # ---------------------------------------------------------
    # Result metadata
    # ---------------------------------------------------------

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{variant}_result.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "variant": variant,

                "test_subject":
                    TEST_SUBJECT,

                "labels":
                    labels,

                "number_of_classes":
                    N_CLASSES,

                "rest_removed":
                    True,

                "train_subjects":
                    sorted(train_subjects),

                "validation_subjects":
                    sorted(val_subjects),

                "epochs":
                    N_EPOCHS,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "temperature":
                    TEMPERATURE,

                "training_mode":
                    training_mode,

                "best_epoch":
                    best_epoch,

                "best_train_loss":
                    best_train_loss,

                "final_train_loss":
                    history[-1]["train_loss"],

                "validation_accuracy":
                    val_acc,

                "best_checkpoint":
                    best_path,

                "final_checkpoint":
                    final_path,
            },
            f,
            indent=2
        )


# =============================================================================
# RUN
# =============================================================================

run_variant(
    VARIANT
)

print(
    f"\nFinished variant: {VARIANT}",
    flush=True
)