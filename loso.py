import os
import sys
import pickle
import subprocess

PROJECT_DIR = "/home/nvidia/24BCE1793/fignn_cni__project"
DATA_PATH = "/lp-dev/24BCE1793/data/processed/all_graphs_29sub.pkl"
SAVE_ROOT = "/lp-dev/24BCE1793/outputs/loso"

GPUS = [1, 2, 5, 6]
SUBJECTS = list(range(1, 30))

N_EPOCHS = 500
BATCH_SIZE = 128

os.makedirs(SAVE_ROOT, exist_ok=True)


def run_fold(subject, gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)

    from src.training.train import run_training

    print("\n" + "=" * 60, flush=True)
    print(f"LOSO FOLD: SUBJECT {subject:02d}", flush=True)
    print(f"GPU: {gpu}", flush=True)
    print("=" * 60, flush=True)

    with open(DATA_PATH, "rb") as f:
        graphs = pickle.load(f)

    # Same Stage-1 setup: remove REST
    graphs = [
        g for g in graphs
        if g.y.item() != 3
    ]

    # Strict LOSO split
    train_graphs = [
        g for g in graphs
        if int(g.subject) != subject
    ]

    test_graphs = [
        g for g in graphs
        if int(g.subject) == subject
    ]

    print(
        f"Train subjects: "
        f"{sorted(set(int(g.subject) for g in train_graphs))}",
        flush=True
    )

    print(
        f"Held-out subject: {subject}",
        flush=True
    )

    print(
        f"Train graphs: {len(train_graphs)}",
        flush=True
    )

    print(
        f"Test graphs: {len(test_graphs)}",
        flush=True
    )

    fold_dir = os.path.join(
        SAVE_ROOT,
        f"subject_{subject:02d}"
    )

    os.makedirs(fold_dir, exist_ok=True)

    # Reuse your existing Stage-1 training
    run_training(
        train_graphs,
        test_graphs=test_graphs,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        save_dir=fold_dir
    )

    print(
        f"\nSUBJECT {subject:02d} COMPLETE",
        flush=True
    )


def main():

    # Run 4 folds concurrently.
    # Then wait for all 4 to finish before launching the next group.

    remaining = SUBJECTS.copy()

    while remaining:

        current = []

        for gpu in GPUS:

            if not remaining:
                break

            subject = remaining.pop(0)
            current.append((subject, gpu))

        processes = []

        for subject, gpu in current:

            print(
                f"Launching Subject {subject:02d} "
                f"on GPU {gpu}",
                flush=True
            )

            log_path = os.path.join(
                SAVE_ROOT,
                f"subject_{subject:02d}.log"
            )

            log_file = open(log_path, "w")

            p = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    str(subject),
                    str(gpu)
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )

            processes.append(
                (subject, p, log_file)
            )

        # Wait for this group of 4
        for subject, p, log_file in processes:

            code = p.wait()
            log_file.close()

            if code != 0:
                print(
                    f"ERROR: Subject {subject:02d} "
                    f"failed with code {code}",
                    flush=True
                )
            else:
                print(
                    f"Subject {subject:02d} finished.",
                    flush=True
                )

    print("\n" + "=" * 60)
    print("ALL 29 LOSO FOLDS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":

    if "--worker" in sys.argv:

        subject = int(sys.argv[2])
        gpu = int(sys.argv[3])

        run_fold(subject, gpu)

    else:

        main()