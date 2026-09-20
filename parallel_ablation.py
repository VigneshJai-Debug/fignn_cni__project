import os
import subprocess
import sys
import time


PROJECT_ROOT = "/home/nvidia/24BCE1793/fignn_cni__project"
OUTPUT_DIR = "/lp-dev/24BCE1793/outputs/ablation_logs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# PRE-SPECIFIED ABLATION FOLDS
# ============================================================

SUBJECTS = [
    7,
    15,
    22,
    29,
]


# ============================================================
# ABLATION VARIANTS
# ============================================================

VARIANTS = [
    "full",
    "no_plv",
    "no_band_weight",
    "node_feature_mlp",
    "gcn",
    "no_supcon",
    "band_power",
]


# ============================================================
# GPUs
# One process per GPU at a time.
# ============================================================

GPUS = [
    "3",
    "0",
    "1",
    "2",
    "5",
    "6",
]


# ============================================================
# BUILD JOB QUEUE
# ============================================================

jobs = []

for subject in SUBJECTS:

    for variant in VARIANTS:

        jobs.append(
            (
                subject,
                variant,
            )
        )


print("=" * 80)
print("CONTROLLED 4-FOLD ABLATION")
print("=" * 80)

print(
    f"Subjects: {SUBJECTS}",
    flush=True,
)

print(
    f"Variants: {VARIANTS}",
    flush=True,
)

print(
    f"Total jobs: {len(jobs)}",
    flush=True,
)

print(
    f"GPUs: {GPUS}",
    flush=True,
)

print(
    "One job per GPU at a time.",
    flush=True,
)

print("=" * 80)


# ============================================================
# ACTIVE PROCESSES
#
# gpu -> (subject, variant, process, log_file)
# ============================================================

active = {}


# ============================================================
# START A JOB
# ============================================================

def start_job(
    subject,
    variant,
    gpu,
):

    log_name = (
        f"subject_{subject:02d}_"
        f"{variant}.log"
    )

    log_path = os.path.join(
        OUTPUT_DIR,
        log_name,
    )

    print(
        f"[START] "
        f"Subject {subject:02d} | "
        f"{variant:<18} | "
        f"GPU {gpu}",
        flush=True,
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["ABLATION_SUBJECT"] = str(subject)
    env["ABLATION_VARIANT"] = variant
    env["PYTHONUNBUFFERED"] = "1"

    log_file = open(
        log_path,
        "w",
        buffering=1,
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "ablation_train.py",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )

    active[gpu] = (
        subject,
        variant,
        process,
        log_file,
    )


# ============================================================
# JOB QUEUE
# ============================================================

pending_jobs = jobs.copy()


# ============================================================
# MAIN SCHEDULER
# ============================================================

while pending_jobs or active:

    # --------------------------------------------------------
    # Fill every free GPU
    # --------------------------------------------------------

    for gpu in GPUS:

        if gpu in active:
            continue

        if not pending_jobs:
            break

        subject, variant = pending_jobs.pop(0)

        start_job(
            subject,
            variant,
            gpu,
        )

        # Small stagger between launches.
        time.sleep(3)


    # --------------------------------------------------------
    # Check running processes
    # --------------------------------------------------------

    finished_gpus = []

    for gpu, info in list(active.items()):

        subject, variant, process, log_file = info

        return_code = process.poll()

        if return_code is None:
            continue

        log_file.close()

        if return_code == 0:

            print(
                f"[DONE ] "
                f"Subject {subject:02d} | "
                f"{variant:<18} | "
                f"GPU {gpu} | "
                f"return code 0",
                flush=True,
            )

        else:

            print(
                f"[FAIL ] "
                f"Subject {subject:02d} | "
                f"{variant:<18} | "
                f"GPU {gpu} | "
                f"return code {return_code}",
                flush=True,
            )

        finished_gpus.append(gpu)


    # --------------------------------------------------------
    # Free completed GPUs
    # --------------------------------------------------------

    for gpu in finished_gpus:
        del active[gpu]


    # --------------------------------------------------------
    # Avoid busy polling
    # --------------------------------------------------------

    if active:
        time.sleep(10)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("ALL ABLATION JOBS FINISHED")
print("=" * 80)

print(
    f"Total jobs completed/attempted: {len(jobs)}",
    flush=True,
)

print(
    f"Logs: {OUTPUT_DIR}",
    flush=True,
)

print("=" * 80)