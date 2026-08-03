"""
evaluate.py
-----------
Generates Stage 1 validation outputs:
  1. t-SNE plot         - latent space coloured by load level
  2. Silhouette score   - cluster quality number
  3. Band importance    - which band the model weighted most
  4. Loss curve         - training behaviour over epochs
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# LABEL_NAMES  = {0: "zeroBACK (rest)", 1: "oneBACK (low)", 2: "twoBACK (high)", 3: "RS eyes closed"}
# LABEL_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336", 3: "#2196F3"}

LABEL_NAMES  = {0: "zeroBACK (rest)", 1: "oneBACK (low)", 2: "twoBACK (high)"}
LABEL_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

def plot_tsne(
    z,
    labels,
    save_path="/lp-dev/24BCE1793/outputs/plots/tsne_latent_space.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Running t-SNE (~30s)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    for lv, ln in LABEL_NAMES.items():
        mask = labels == lv
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=LABEL_COLORS[lv], label=ln,
                   alpha=0.55, s=18, edgecolors="none")
    ax.set_title("FIGNN Stage 1 - Latent Space (t-SNE)", color="white", fontsize=13, pad=14)
    ax.set_xlabel("t-SNE dim 1", color="#aaaaaa")
    ax.set_ylabel("t-SNE dim 2", color="#aaaaaa")
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(framealpha=0.2, facecolor="#222244", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return z_2d

def compute_silhouette(z, labels):
    score = silhouette_score(z, labels)
    print(f"Silhouette score: {score:.4f}")
    return score

def plot_band_importance(
    band_weights,
    save_path="/lp-dev/24BCE1793/outputs/plots/band_importance.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bands  = list(band_weights.keys())
    values = list(band_weights.values())
    colors = ["#7986CB", "#26C6DA", "#66BB6A", "#FFA726"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    bars = ax.bar(bands, values, color=colors, width=0.5)
    ax.set_ylabel("Learned weight (softmax)", color="#aaaaaa")
    ax.set_title("Band Importance Learned by FIGNN", color="white", fontsize=12)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", color="white", fontsize=9)
    ax.set_ylim(0, max(values) * 1.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_loss_curve(
    train_history,
    val_history,
    save_path="/lp-dev/24BCE1793/outputs/plots/loss_curve.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_epochs = list(range(1, len(train_history) + 1))
    val_epochs = [1] + list(range(10, len(train_history) + 1, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#12122a")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
    ax1.plot(train_epochs, train_history, color="#26C6DA", linewidth=1.5)
    ax1.set_title("Training Loss", color="white")
    ax1.set_xlabel("Epoch", color="#aaaaaa")
    ax1.set_ylabel("SupCon Loss", color="#aaaaaa")
    ax2.plot(
        val_epochs[:len(val_history)],
        val_history,
        color="#66BB6A",
        linewidth=1.5,
        marker="o"
    )
    ax2.set_title("Validation Silhouette", color="white")
    ax2.set_xlabel("Epoch", color="#aaaaaa")
    ax2.set_ylabel("Silhouette", color="#aaaaaa")
    ax2.axhline(0.3, color="#FFA726", linestyle="--", alpha=0.5, label="Target")
    ax2.legend(labelcolor="white", facecolor="#222244", framealpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def run_full_evaluation(model, graph_dataset, device, batch_size=32):
    from torch_geometric.loader import DataLoader
    from src.training.train import evaluate
    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    z, labels = evaluate(model, loader, device)
    sil = compute_silhouette(z, labels)
    plot_tsne(z, labels)
    bw = model.get_band_importance()
    print(f"PLV band weights:  {bw['plv']}")
    print(f"Corr band weights: {bw['corr']}")
    # plot both separately
    plot_band_importance(
        bw["plv"],
        save_path="/lp-dev/24BCE1793/outputs/plots/band_importance_plv.png"
    )
    
    plot_band_importance(
        bw["corr"],
        save_path="/lp-dev/24BCE1793/outputs/plots/band_importance_corr.png"
    )
    return {"silhouette": sil, "band_weights": bw, "z": z, "labels": labels}