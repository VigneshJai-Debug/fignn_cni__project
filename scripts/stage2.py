"""Stage 2 — 64-D unseen-subject calibration comparison.

Methods:
    population_only, midpoint, regression, frozen_geometric.

Each LOSO fold uses the authoritative fold-specific fignn_best.pth and
normalisation_stats.pt. Errors are direct Euclidean distances in the original
64-D latent space.
"""

import os
import pickle
import copy
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, ttest_rel
from torch_geometric.loader import DataLoader

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPHS_PATH = (
    PROJECT_ROOT / "data" / "data" / "processed" / "all_graphs_29sub.pkl"
)
LOSO_ROOT = PROJECT_ROOT / "data" / "outputs" / "loso"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "outputs" / "stage2_calibration_comparison_64D.csv"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model import FIGNNEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METHODS = ["population_only", "midpoint", "regression", "frozen_geometric"]

print("Loading graphs...")
with open(GRAPHS_PATH, "rb") as f:
    all_graphs = pickle.load(f)
print(f"Loaded {len(all_graphs)} graphs")
print(f"Using device: {DEVICE}")


def load_fold_normalisation(test_subject):
    stats_path = os.path.join(LOSO_ROOT, f"subject_{test_subject:02d}", "normalisation_stats.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(stats_path)
    stats = torch.load(stats_path, map_location="cpu")
    return stats["mean"].float(), stats["std"].float().clamp(min=1e-8)


def load_fold_model(test_subject):
    checkpoint_path = os.path.join(LOSO_ROOT, f"subject_{test_subject:02d}", "fignn_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    model = FIGNNEncoder().to(DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, checkpoint_path


@torch.no_grad()
def get_embeddings(model, graphs, mean, std, device=DEVICE, batch_size=128):
    processed = []
    for g in graphs:
        g_copy = copy.copy(g)
        g_copy.x = (g.x.clone() - mean) / std
        processed.append(g_copy)

    loader = DataLoader(processed, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_z = []
    all_y = []

    for batch in loader:
        batch = batch.to(device)
        z = model(batch)
        all_z.append(z.cpu().numpy())
        all_y.append(batch.y.view(-1).cpu().numpy())

    return np.vstack(all_z), np.concatenate(all_y)


def get_condition_centroids(z, y):
    centroids = {}

    for label, name in [(3, "REST"), (0, "0B"), (1, "1B"), (2, "2B")]:
        mask = y == label

        if not np.any(mask):
            raise ValueError(f"No samples found for condition {name}")

        centroids[name] = z[mask].mean(axis=0)

    return centroids


def projection_position(rest, target, two_back):
    direction = two_back - rest
    denom = np.dot(direction, direction)
    if denom < 1e-12:
        return np.nan
    return float(np.dot(target - rest, direction) / denom)


def population_only_prediction(train_centroids):
    return np.mean([c["1B"] for c in train_centroids], axis=0)


def midpoint_prediction(true_rest, true_two_back):
    return true_rest + 0.5 * (true_two_back - true_rest)


def regression_prediction(train_centroids, true_rest, true_two_back):
    x = np.asarray([np.concatenate([c["REST"], c["2B"]]) for c in train_centroids], dtype=np.float64)
    y = np.asarray([c["1B"] for c in train_centroids], dtype=np.float64)
    x_test = np.concatenate([true_rest, true_two_back]).astype(np.float64)
    x_aug = np.column_stack([np.ones(len(x)), x])
    test_aug = np.concatenate([[1.0], x_test])
    coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    return test_aug @ coef


def frozen_geometric_prediction(train_centroids, true_rest, true_two_back):
    alpha_values = [projection_position(c["REST"], c["1B"], c["2B"]) for c in train_centroids]
    alpha_values = [a for a in alpha_values if np.isfinite(a)]
    if not alpha_values:
        raise ValueError("Could not calculate population alpha.")
    alpha_population = float(np.mean(alpha_values))
    return true_rest + alpha_population * (true_two_back - true_rest), alpha_population

def main():
    results = []
    
    for test_subject in range(1, 30):
        print("\n" + "=" * 70)
        print(f"Processing unseen Subject {test_subject:02d}")
        print("=" * 70)
    
        train_graphs = [g for g in all_graphs if int(g.subject) != test_subject]
        test_graphs = [g for g in all_graphs if int(g.subject) == test_subject]
        train_subjects = sorted({int(g.subject) for g in train_graphs})
        if len(train_subjects) != 28:
            raise ValueError(f"Expected 28 training subjects, got {len(train_subjects)}")
    
        mean, std = load_fold_normalisation(test_subject)
        model, checkpoint_path = load_fold_model(test_subject)
    
        train_subject_centroids = []
        for sid in train_subjects:
            sid_graphs = [g for g in train_graphs if int(g.subject) == sid]
            z_sid, y_sid = get_embeddings(model, sid_graphs, mean, std)
            train_subject_centroids.append(get_condition_centroids(z_sid, y_sid))
    
        z_test, y_test = get_embeddings(model, test_graphs, mean, std)
        test_centroids = get_condition_centroids(z_test, y_test)
        true_rest = test_centroids["REST"]
        true_1b = test_centroids["1B"]
        true_2b = test_centroids["2B"]
    
        predictions = {
            "population_only": population_only_prediction(train_subject_centroids),
            "midpoint": midpoint_prediction(true_rest, true_2b),
            "regression": regression_prediction(train_subject_centroids, true_rest, true_2b),
        }
        frozen_prediction, alpha_population = frozen_geometric_prediction(
            train_subject_centroids, true_rest, true_2b
        )
        predictions["frozen_geometric"] = frozen_prediction
    
        for method, prediction in predictions.items():
            prediction = np.asarray(prediction, dtype=np.float64)
            error = float(np.linalg.norm(prediction - true_1b))
            alpha = projection_position(true_rest, prediction, true_2b) if method in {"midpoint", "frozen_geometric"} else np.nan
            results.append({
                "subject": test_subject,
                "method": method,
                "error_64D": error,
                "prediction_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                "population_alpha": alpha_population if method == "frozen_geometric" else np.nan,
                "prediction_norm": float(np.linalg.norm(prediction)),
                "true_1B_norm": float(np.linalg.norm(true_1b)),
                "rest_norm": float(np.linalg.norm(true_rest)),
                "two_back_norm": float(np.linalg.norm(true_2b)),
                "checkpoint": checkpoint_path,
            })
            print(f"{method:20s} | 64-D error = {error:.6f}")
    
    stage2 = pd.DataFrame(results).sort_values(["subject", "method"]).reset_index(drop=True)
    stage2.to_csv(OUTPUT_PATH, index=False)
    
    summary = stage2.groupby("method")["error_64D"].agg(mean="mean", std="std", median="median", count="count").reindex(METHODS)
    print("\n" + "=" * 70)
    print("STAGE 2 — 64-D METHOD SUMMARY")
    print("=" * 70)
    print(summary.round(6).to_string())
    
    baseline = stage2[stage2["method"] == "population_only"][["subject", "error_64D"]].rename(columns={"error_64D": "population_error_64D"})
    comparison = stage2.merge(baseline, on="subject", how="left")
    comparison["absolute_improvement"] = comparison["population_error_64D"] - comparison["error_64D"]
    comparison["percent_improvement"] = np.where(comparison["population_error_64D"] > 1e-12, 100.0 * comparison["absolute_improvement"] / comparison["population_error_64D"], np.nan)
    comparison["improved_over_population"] = comparison["error_64D"] < comparison["population_error_64D"]
    
    print("\n" + "=" * 70)
    print("COMPARISON AGAINST POPULATION-ONLY")
    print("=" * 70)
    print(comparison[["subject", "method", "error_64D", "population_error_64D", "absolute_improvement", "percent_improvement", "improved_over_population"]].sort_values(["subject", "method"]).round(6).to_string(index=False))
    
    test_rows = []
    for method in METHODS[1:]:
        base = stage2[stage2["method"] == "population_only"].sort_values("subject")
        current = stage2[stage2["method"] == method].sort_values("subject")
        base_err = base["error_64D"].to_numpy()
        method_err = current["error_64D"].to_numpy()
        w_stat, w_p = wilcoxon(base_err, method_err, alternative="greater")
        t_stat, t_p = ttest_rel(base_err, method_err)
        diffs = base_err - method_err
        sd_diff = float(np.std(diffs, ddof=1))
        test_rows.append({
            "method": method,
            "wilcoxon_W": float(w_stat),
            "wilcoxon_p": float(w_p),
            "paired_t_t": float(t_stat),
            "paired_t_p": float(t_p),
            "mean_error_reduction": float(np.mean(diffs)),
            "cohen_dz": float(np.mean(diffs) / sd_diff) if sd_diff > 1e-12 else np.nan,
            "n_improved": int(np.sum(method_err < base_err)),
            "n_subjects": len(base_err),
        })
    
    tests = pd.DataFrame(test_rows)
    order = np.argsort(tests["wilcoxon_p"].to_numpy())
    adjusted = np.empty(len(tests))
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (len(tests) - rank) * tests.loc[idx, "wilcoxon_p"])
        running = max(running, value)
        adjusted[idx] = running
    tests["wilcoxon_p_holm"] = adjusted
    
    print("\n" + "=" * 70)
    print("PAIRED STATISTICAL TESTS VS POPULATION-ONLY")
    print("=" * 70)
    print(tests.round(8).to_string(index=False))
    
    tests.to_csv(OUTPUT_PATH.replace(".csv", "_paired_tests.csv"), index=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Saved paired tests to: {OUTPUT_PATH.replace('.csv', '_paired_tests.csv')}")
if __name__ == "__main__":
    main()