import sys, os, pickle
sys.path.insert(0, '/home/nvidia/24BCE1793/fignn_cni__project')
os.chdir('/home/nvidia/24BCE1793/fignn_cni__project')
os.makedirs("/lp-dev/24BCE1793/data/processed", exist_ok=True)

from src.preprocessing.preprocess import process_file
from src.preprocessing.plv import batch_compute_plv
from src.preprocessing.graph_builder import build_graph_dataset

DATA_ROOT = "/lp-dev/24BCE1793/data/raw"
TASKS = {
    "zeroBACK":  0,
    "oneBACK":   1,
    "twoBACK":   2,
    "RS_Beg_EC": 3,
}
SESSIONS  = ["ses-S1", "ses-S2", "ses-S3"]
SUBJECTS  = [f"sub-{i:02d}" for i in range(1, 30)]

set_files = []
for sub in SUBJECTS:
    for ses in SESSIONS:
        for task, label in TASKS.items():
            path = os.path.join(DATA_ROOT, sub, ses, "eeg", f"{task}.set")
            if os.path.exists(path):
                set_files.append((path, label))
            else:
                print(f"[skip] {sub}/{ses}/{task}", flush=True)

print(f"Found {len(set_files)} files", flush=True)

all_graphs = []
for i, (set_path, label) in enumerate(set_files):
    sub  = set_path.split('/')[-4]
    task = set_path.split('/')[-1]
    print(f"[{i+1}/{len(set_files)}] {sub} {task}", flush=True)
    try:
        processed = process_file(set_path, label)
        plv_list  = batch_compute_plv(processed["band_epochs"], verbose=False)
        graphs    = build_graph_dataset(processed, plv_list)
        all_graphs.extend(graphs)
        print(f"  +{len(graphs)} graphs | total: {len(all_graphs)}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e} — skipping", flush=True)
        continue

print(f"\nDataset: {len(all_graphs)} graphs", flush=True)

# Label distribution
for lv, ln in [(0,"zeroBACK"),(1,"oneBACK"),(2,"twoBACK"),(3,"RS_Beg_EC")]:
    c = sum(1 for g in all_graphs if g.y.item()==lv)
    print(f"  label {lv} ({ln}): {c} graphs", flush=True)

# Check edge count on sample graph
sample_edges = all_graphs[0].edge_index.shape[1]
print(f"\nSample graph edges: {sample_edges} (was ~2400 before, should be lower now)", flush=True)

with open("/lp-dev/24BCE1793/data/processed/all_graphs_29sub.pkl", "wb") as f:
    pickle.dump(all_graphs, f)
print("Saved: all_graphs_29sub.pkl", flush=True)
