# ============================================================
# STAGE 2 — 64-D UNSEEN-SUBJECT CALIBRATION COMPARISON
#
# Question:
# Does subject-specific REST + 2-back calibration improve
# estimation of an unseen subject's true 1-back latent centroid?
#
# Comparison:
#
# BASELINE:
#     Predict unseen subject's 1-back using only the population
#     1-back centroid from the 28 training subjects.
#
# CALIBRATED:
#     Use unseen subject's REST and 2-back centroids as personal
#     anchors, and place predicted 1-back at the population-derived
#     relative position along that subject's REST → 2-back vector.
#
# Evaluation:
#     Direct Euclidean error in original 64-D latent space.
# ============================================================


import os
import re
import glob
import pickle
import copy

import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

import sys


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/lp-dev/24BCE1793/fignn_cni__project"

GRAPHS_PATH = (
    "/lp-dev/24BCE1793/data/processed/"
    "all_graphs_29sub.pkl"
)

LOSO_ROOT = (
    "/lp-dev/24BCE1793/outputs/loso"
)

OUTPUT_PATH = (
    "/lp-dev/24BCE1793/outputs/"
    "stage2_calibration_comparison_64D.csv"
)


sys.path.insert(0, PROJECT_ROOT)

from src.models.model import FIGNNEncoder


# ============================================================
# LOAD ALL GRAPHS
# ============================================================

print("Loading graphs...")

with open(GRAPHS_PATH, "rb") as f:
    all_graphs = pickle.load(f)

print(f"Loaded {len(all_graphs)} graphs")


# ============================================================
# FIND LOSO CHECKPOINTS
# ============================================================

checkpoint_files = glob.glob(
    os.path.join(
        LOSO_ROOT,
        "**",
        "fignn_final.pth"
    ),
    recursive=True
)

if len(checkpoint_files) == 0:
    raise FileNotFoundError(
        f"No LOSO checkpoints found under:\n{LOSO_ROOT}"
    )


subject_checkpoints = {}

for path in checkpoint_files:

    match = re.search(
        r"subject[_-]?(\d+)",
        path,
        re.IGNORECASE
    )

    if match:

        subject = int(match.group(1))

        subject_checkpoints[subject] = path


print(
    f"\nFound {len(subject_checkpoints)} "
    f"LOSO checkpoints"
)


# ============================================================
# COMPUTE NORMALISATION STATISTICS
#
# IMPORTANT:
# Each LOSO fold gets statistics computed ONLY from its
# 28 training subjects.
#
# This avoids leakage from the unseen test subject.
# ============================================================

def compute_train_stats(graphs):

    all_x = torch.cat(
        [g.x for g in graphs],
        dim=0
    )

    mean = all_x.mean(dim=0)

    std = (
        all_x.std(dim=0)
        .clamp(min=1e-8)
    )

    return mean, std


# ============================================================
# EXTRACT 64-D EMBEDDINGS
# ============================================================

@torch.no_grad()
def get_embeddings(
    model,
    graphs,
    mean,
    std,
    device,
    batch_size=128
):

    processed = []

    for g in graphs:

        g_copy = copy.copy(g)

        g_copy.x = (
            g.x.clone() - mean
        ) / std

        processed.append(g_copy)


    loader = DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=False
    )


    model.eval()

    all_z = []
    all_y = []


    for batch in loader:

        batch = batch.to(device)

        z = model(batch)

        all_z.append(
            z.cpu().numpy()
        )

        all_y.append(
            batch.y
            .view(-1)
            .cpu()
            .numpy()
        )


    return (
        np.vstack(all_z),
        np.concatenate(all_y)
    )


# ============================================================
# CONDITION CENTROIDS
# ============================================================

def get_condition_centroids(z, y):

    centroids = {}

    for label, name in [
        (3, "REST"),
        (0, "0B"),
        (1, "1B"),
        (2, "2B")
    ]:

        condition_z = z[y == label]

        if len(condition_z) == 0:

            raise ValueError(
                f"No samples found for {name}"
            )

        centroids[name] = (
            condition_z.mean(axis=0)
        )


    return centroids


# ============================================================
# POPULATION RELATIVE 1-BACK POSITION
#
# For each training subject:
#
#     REST -----> 2B
#
# We project that subject's 1B centroid onto their
# REST → 2B direction.
#
# alpha = 0:
#     exactly REST
#
# alpha = 1:
#     exactly 2-back
#
# alpha between:
#     intermediate workload position
#
# We then take the mean alpha across the 28 training subjects.
# ============================================================

def projection_position(
    rest,
    target,
    two_back
):

    direction = (
        two_back - rest
    )

    numerator = np.dot(
        target - rest,
        direction
    )

    denominator = np.dot(
        direction,
        direction
    )

    if denominator < 1e-12:
        return np.nan


    alpha = (
        numerator / denominator
    )

    return alpha


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# MAIN LOSO STAGE 2
# ============================================================

results = []


for test_subject in sorted(subject_checkpoints):


    print(
        f"\n{'='*65}"
    )

    print(
        f"Processing unseen Subject "
        f"{test_subject}"
    )

    print(
        f"{'='*65}"
    )


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_graphs = [

        g for g in all_graphs

        if g.subject != test_subject

    ]


    test_graphs = [

        g for g in all_graphs

        if g.subject == test_subject

    ]


    train_subjects = sorted(
        set(
            g.subject
            for g in train_graphs
        )
    )


    # --------------------------------------------------------
    # COMPUTE FOLD-SPECIFIC NORMALISATION
    #
    # ONLY training subjects
    # --------------------------------------------------------

    mean, std = compute_train_stats(
        train_graphs
    )


    # --------------------------------------------------------
    # LOAD FROZEN LOSO MODEL
    # --------------------------------------------------------

    checkpoint_path = (
        subject_checkpoints[
            test_subject
        ]
    )


    model = (
        FIGNNEncoder()
        .to(device)
    )


    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    # --------------------------------------------------------
    # GENERATE TRAINING POPULATION EMBEDDINGS
    #
    # REST + 0B + 1B + 2B
    #
    # The model itself was trained only on task classes,
    # but REST is passed through the frozen encoder here
    # purely as a calibration anchor.
    # --------------------------------------------------------

    train_z, train_y = get_embeddings(

        model,
        train_graphs,
        mean,
        std,
        device

    )


    # --------------------------------------------------------
    # GET EACH TRAIN SUBJECT'S CONDITION CENTROIDS
    #
    # We need subject-level centroids to calculate the
    # population-relative position of 1B between REST and 2B.
    # --------------------------------------------------------

    train_subject_centroids = []


    for subject in train_subjects:


        subject_graphs = [

            g for g in train_graphs

            if g.subject == subject

        ]


        z_sub, y_sub = get_embeddings(

            model,
            subject_graphs,
            mean,
            std,
            device

        )


        centroids = (
            get_condition_centroids(
                z_sub,
                y_sub
            )
        )


        train_subject_centroids.append(
            centroids
        )


    # --------------------------------------------------------
    # POPULATION 1-BACK CENTROID
    #
    # BASELINE PREDICTION:
    #
    # No personal information from unseen subject.
    # --------------------------------------------------------

    population_1B = np.mean(

        [
            c["1B"]

            for c in
            train_subject_centroids
        ],

        axis=0

    )


    # --------------------------------------------------------
    # LEARN POPULATION RELATIVE POSITION OF 1B
    #
    # alpha_i =
    #
    # projection of subject i's 1B
    # onto their REST → 2B direction
    #
    # Then:
    #
    # alpha_population = mean(alpha_i)
    # --------------------------------------------------------

    alpha_values = []


    for c in train_subject_centroids:


        alpha = projection_position(

            c["REST"],
            c["1B"],
            c["2B"]

        )


        if not np.isnan(alpha):

            alpha_values.append(
                alpha
            )


    alpha_population = float(
        np.mean(alpha_values)
    )


    alpha_median = float(
        np.median(alpha_values)
    )


    # --------------------------------------------------------
    # OPTIONAL:
    #
    # Also compute median-alpha calibrated prediction.
    #
    # Mean alpha is the primary result.
    # Median is useful because alpha can occasionally
    # be affected by outlier subjects.
    # --------------------------------------------------------


    # ========================================================
    # UNSEEN SUBJECT
    # ========================================================

    test_z, test_y = get_embeddings(

        model,
        test_graphs,
        mean,
        std,
        device

    )


    test_centroids = (
        get_condition_centroids(
            test_z,
            test_y
        )
    )


    true_rest = (
        test_centroids["REST"]
    )

    true_1B = (
        test_centroids["1B"]
    )

    true_2B = (
        test_centroids["2B"]
    )


    # ========================================================
    # BASELINE — NO CALIBRATION
    # ========================================================

    baseline_prediction = (
        population_1B
    )


    # ========================================================
    # CALIBRATED PREDICTION
    #
    # predicted_1B =
    #
    # REST_subject
    # +
    # alpha_population
    # *
    # (2B_subject - REST_subject)
    #
    # This produces a complete 64-D predicted vector.
    # ========================================================

    calibrated_prediction = (

        true_rest

        +

        alpha_population
        *
        (
            true_2B
            -
            true_rest
        )

    )


    # --------------------------------------------------------
    # MEDIAN VERSION
    # --------------------------------------------------------

    calibrated_prediction_median = (

        true_rest

        +

        alpha_median
        *
        (
            true_2B
            -
            true_rest
        )

    )


    # ========================================================
    # 64-D EUCLIDEAN ERRORS
    # ========================================================

    baseline_error = float(

        np.linalg.norm(

            baseline_prediction
            -
            true_1B

        )

    )


    calibrated_error = float(

        np.linalg.norm(

            calibrated_prediction
            -
            true_1B

        )

    )


    calibrated_error_median = float(

        np.linalg.norm(

            calibrated_prediction_median
            -
            true_1B

        )

    )


    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    absolute_improvement = (

        baseline_error
        -
        calibrated_error

    )


    if baseline_error > 1e-12:

        percent_improvement = (

            absolute_improvement
            /
            baseline_error

        ) * 100

    else:

        percent_improvement = np.nan


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "Subject":
            test_subject,

        "population_alpha_mean":
            alpha_population,

        "population_alpha_median":
            alpha_median,


        "baseline_error_64D":
            baseline_error,

        "calibrated_error_64D":
            calibrated_error,

        "calibrated_error_median_64D":
            calibrated_error_median,


        "absolute_improvement":
            absolute_improvement,

        "percent_improvement":
            percent_improvement,


        "calibration_helped":
            calibrated_error
            <
            baseline_error,

        "baseline_prediction_norm":
            float(
                np.linalg.norm(
                    baseline_prediction
                )
            ),

        "true_1B_norm":
            float(
                np.linalg.norm(
                    true_1B
                )
            ),

        "calibrated_prediction_norm":
            float(
                np.linalg.norm(
                    calibrated_prediction
                )
            )

    })


    # ========================================================
    # PRINT SUBJECT RESULT
    # ========================================================

    print(

        f"Population alpha: "
        f"{alpha_population:.6f}"

    )

    print(

        f"Baseline error:   "
        f"{baseline_error:.6f}"

    )

    print(

        f"Calibrated error: "
        f"{calibrated_error:.6f}"

    )

    print(

        f"Improvement:      "
        f"{percent_improvement:.2f}%"

    )

    print(

        f"Calibration helped: "
        f"{calibrated_error < baseline_error}"

    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values("Subject")
    .reset_index(drop=True)
)


print(
    "\n\n"
    +
    "=" * 70
)

print(
    "STAGE 2 — CALIBRATION COMPARISON RESULTS"
)

print(
    "=" * 70
)


print(
    results_df.round(6).to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

n_subjects = len(
    results_df
)


n_helped = int(

    results_df[
        "calibration_helped"
    ].sum()

)


mean_baseline = float(

    results_df[
        "baseline_error_64D"
    ].mean()

)


mean_calibrated = float(

    results_df[
        "calibrated_error_64D"
    ].mean()

)


median_baseline = float(

    results_df[
        "baseline_error_64D"
    ].median()

)


median_calibrated = float(

    results_df[
        "calibrated_error_64D"
    ].median()

)


overall_improvement = (

    (
        mean_baseline
        -
        mean_calibrated
    )

    /

    mean_baseline

    * 100

)


print(
    f"\nSubjects tested: "
    f"{n_subjects}"
)


print(
    f"\nCalibration improved prediction for: "
    f"{n_helped} / {n_subjects} subjects"
)


print(
    f"\nMean baseline error:   "
    f"{mean_baseline:.6f}"
)


print(
    f"Mean calibrated error: "
    f"{mean_calibrated:.6f}"
)


print(
    f"\nMedian baseline error:   "
    f"{median_baseline:.6f}"
)


print(
    f"Median calibrated error: "
    f"{median_calibrated:.6f}"
)


print(
    f"\nOverall mean error improvement: "
    f"{overall_improvement:.2f}%"
)


# ============================================================
# PAIRED STATISTICAL TEST
# ============================================================

try:

    from scipy.stats import (
        wilcoxon
    )


    stat, p_value = wilcoxon(

        results_df[
            "baseline_error_64D"
        ],

        results_df[
            "calibrated_error_64D"
        ],

        alternative="greater"

    )


    print(
        "\nWilcoxon signed-rank test"
    )

    print(
        "H1: baseline error > calibrated error"
    )

    print(
        f"Statistic: {stat:.6f}"
    )

    print(
        f"p-value:   {p_value:.8f}"
    )


except Exception as e:

    print(
        f"\nCould not run Wilcoxon test: {e}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(

    OUTPUT_PATH,

    index=False

)


print(
    f"\nSaved results to:\n"
    f"{OUTPUT_PATH}"
)