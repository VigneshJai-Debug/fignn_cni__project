import os
import pickle
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"/ "data" / "raw"
OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "data"
    / "processed"
    / "pre_graph_data_reconstructed.pkl"
)

from src.preprocessing.preprocess import process_file
from src.preprocessing.plv import batch_compute_plv
from src.preprocessing.graph_builder import compute_band_power

BANDS = ["delta", "theta", "alpha", "beta"]

TASKS = {
    "zeroBACK": 0,
    "oneBACK": 1,
    "twoBACK": 2,
}

SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 30)]


def main():

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    set_files = []

    for sub in SUBJECTS:
        for ses in SESSIONS:
            for task, label in TASKS.items():

                path = os.path.join(
                    DATA_ROOT,
                    sub,
                    ses,
                    "eeg",
                    f"{task}.set"
                )

                if os.path.exists(path):
                    set_files.append((path, label))
                else:
                    print(f"[skip] {sub}/{ses}/{task}")

    print(f"Found {len(set_files)} task files")

    pre_graph_data = []

    for file_idx, (set_path, label) in enumerate(set_files, 1):

        sub_str = set_path.split("/")[-4]
        ses_str = set_path.split("/")[-3]

        subject_id = int(sub_str.split("-")[1])
        session_id = int(ses_str.split("-S")[1])

        print(
            f"\n[{file_idx}/{len(set_files)}] "
            f"sub-{subject_id:02d} ses-S{session_id} "
            f"label={label}"
        )

        # Same preprocessing used by the graph pipeline
        processed = process_file(set_path, label)

        # Same PLV computation used by graph construction
        connectivity_list = batch_compute_plv(
            processed["band_epochs"],
            verbose=False
        )

        n_epochs = processed["n_epochs"]

        print(
            f"  channels={processed['n_channels']} "
            f"epochs={n_epochs}"
        )

        for epoch_idx in range(n_epochs):

            # Same relative band-power node features
            node_features = compute_band_power(
                processed["band_epochs"],
                epoch_idx
            )

            connectivity = connectivity_list[epoch_idx]

            record = {
                "connectivity": {
                    band: connectivity[band].astype(np.float32)
                    for band in BANDS
                },

                "node_features": node_features.astype(np.float32),

                "label": int(label),
                "subject_id": int(subject_id),
                "session_id": int(session_id),
                "epoch_idx": int(epoch_idx),

                "source_file": set_path,
                "n_channels": int(processed["n_channels"]),
                "sfreq": float(processed["sfreq"]),
                "ch_names": list(processed["ch_names"]),
            }

            pre_graph_data.append(record)

        print(
            f"  added {n_epochs} records | "
            f"total={len(pre_graph_data)}"
        )

    print("\n" + "=" * 60)
    print(f"TOTAL RECORDS: {len(pre_graph_data)}")

    # Basic checks
    bad = 0

    for r in pre_graph_data:

        x = r["node_features"]

        if x.ndim != 2 or x.shape[1] != 4:
            bad += 1
            continue

        for band in BANDS:
            mat = r["connectivity"][band]

            if mat.shape != (x.shape[0], x.shape[0]):
                bad += 1
                break

    print(f"BAD RECORDS: {bad}")

    # Label distribution
    for label in [0, 1, 2]:
        count = sum(
            r["label"] == label
            for r in pre_graph_data
        )
        print(f"label {label}: {count}")

    # Save
    print(f"\nSaving to:\n{OUTPUT}")

    with open(OUTPUT, "wb") as f:
        pickle.dump(
            pre_graph_data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    size_gb = os.path.getsize(OUTPUT) / (1024 ** 3)

    print(f"Saved successfully: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()