import sys, os, pickle, torch
sys.path.insert(0, '/home/nvidia/24BCE1793/fignn_cni__project')
os.chdir('/home/nvidia/24BCE1793/fignn_cni__project')
os.makedirs("/lp-dev/24BCE1793/outputs/plots", exist_ok=True)
os.makedirs("/lp-dev/24BCE1793/outputs/checkpoints", exist_ok=True)

print("Loading graphs...", flush=True)
with open("/lp-dev/24BCE1793/data/processed/all_graphs_29sub.pkl", "rb") as f:
    all_graphs = pickle.load(f)
print(f"Loaded {len(all_graphs)} graphs", flush=True)

from src.training.train import run_training
from src.evaluation.evaluate import run_full_evaluation, plot_loss_curve

model, train_hist, val_hist = run_training(
    all_graphs,
    n_epochs   = 500,
    batch_size = 128,    # larger batch = more pairs per SupCon step
    lr         = 1e-3,
    save_dir   = "/lp-dev/24BCE1793/outputs/checkpoints"
)

plot_loss_curve(train_hist, val_hist,
    save_path="/lp-dev/24BCE1793/outputs/plots/loss_curve.png")

device = torch.device("cuda")
results = run_full_evaluation(model, all_graphs, device, batch_size=64)

print(f"\n=== STAGE 1 COMPLETE ===", flush=True)
print(f"Silhouette : {results['silhouette']:.4f}", flush=True)
print(f"Band weights: {results['band_weights']}", flush=True)
