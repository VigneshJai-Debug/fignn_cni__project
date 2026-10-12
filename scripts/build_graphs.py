import sys
import os
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"/ "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "data" / "processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

from src.preprocessing.preprocess import process_file
from src.preprocessing.plv import batch_compute_plv
from src.preprocessing.graph_builder import build_graph_dataset

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
    sub_str = Path(set_path).parts[-4]
    ses_str = Path(set_path).parts[-3]
    sub_id  = int(sub_str.split('-')[1])  # 1
    ses_id  = int(ses_str.split('-S')[1]) # 1

    processed    = process_file(set_path, label)
    conn_list    = batch_compute_plv(processed["band_epochs"], verbose=False)
    graphs       = build_graph_dataset(processed, conn_list, 
                                        subject_id=sub_id, 
                                        session_id=ses_id)
    all_graphs.extend(graphs)

print(f"\nDataset: {len(all_graphs)} graphs", flush=True)

# Label distribution
for lv, ln in [(0,"zeroBACK"),(1,"oneBACK"),(2,"twoBACK"),(3,"RS_Beg_EC")]:
    c = sum(1 for g in all_graphs if g.y.item()==lv)
    print(f"  label {lv} ({ln}): {c} graphs", flush=True)

# Check edge count on sample graph
sample_edges = all_graphs[0].edge_index.shape[1]
print(f"\nSample graph edges: {sample_edges} (was ~2400 before, should be lower now)", flush=True)

with open(PROCESSED_ROOT / "all_graphs_29sub.pkl", "wb") as f:
    pickle.dump(all_graphs, f)
print("Saved: all_graphs_29sub.pkl", flush=True)
