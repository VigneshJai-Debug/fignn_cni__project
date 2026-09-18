import os
import subprocess
import sys
import time

PROJECT_ROOT = "/home/nvidia/24BCE1793/fignn_cni__project"
OUTPUT_DIR = "/lp-dev/24BCE1793/outputs/ablation_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full is included as the matching baseline.
jobs = [
    ("full", "3"),
    ("no_plv", "0"),
    ("no_band_weight", "1"),
    ("no_gat", "2"),
    ("no_supcon", "5"),
    ("band_power", "6"),
]

processes = []

for variant, gpu in jobs:
    log_path = os.path.join(
        OUTPUT_DIR,
        f"{variant}.log",
    )

    print(
        f"Starting {variant} on GPU {gpu}",
        flush=True,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
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

    processes.append(
        (
            variant,
            process,
            log_file,
        )
    )

    # Small delay so the processes do not initialize simultaneously.
    time.sleep(3)

for variant, process, log_file in processes:
    return_code = process.wait()
    log_file.close()

    print(
        f"{variant} finished with return code {return_code}",
        flush=True,
    )