# sweep.py
import os
import re
import time
import subprocess
import json
import csv

# Define the scaled-down protocol parameters
DATASET = "MNIST"
SIZES = [2, 4, 6]       # Small, medium, and intermediate triggers
SEEDS = [42, 100, 2026] # 3 random seeds
TARGET_LABEL = 0        # Fixed target label
POISON_RATE = 0.1       # Fixed 10% poison rate

RESULTS_DIR = "./results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
CSV_PATH = os.path.join(RESULTS_DIR, "sweep.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize CSV with both original columns and analyze_results.py columns
columns = [
    "dataset", "model_type", "is_clean", "trigger_size", "seed", "target_label",
    "clean_accuracy", "attack_success_rate",
    "anomaly_index", "true_target_anomaly_index", "max_anomaly_index",
    "flagged_label", "matches_true_target", "runtime_sec"
]

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)


def parse_mad_output(output: str):
    """Parse anomaly index and flagged label from mad_outlier_detection.py stdout."""
    anomaly_index = 0.0
    flagged_label = -1

    for line in output.splitlines():
        lower = line.lower().strip()
        if "anomaly index:" in lower:
            match = re.search(r"anomaly index:\s*([0-9]*\.?[0-9]+)", lower)
            if match:
                anomaly_index = float(match.group(1))

        # Handles both "flagged label: 0" and original "flagged label list: 0: 16.11"
        if "flagged label" in lower:
            after_colon = line.split(":", 1)[-1].strip()
            match = re.search(r"(-?\d+)", after_colon)
            if match:
                flagged_label = int(match.group(1))

    return anomaly_index, flagged_label


def run_experiment(trigger_size, seed, model_type):
    print(f"\n{'='*50}\nRunning {model_type} model: size={trigger_size}, seed={seed}\n{'='*50}")
    is_clean = 1 if model_type == "clean" or trigger_size == 0 else 0
    effective_target = -1 if is_clean else TARGET_LABEL

    # 1. Train the Model
    # ---------------------------------------------------------
    train_cmd = [
        "python", "mnist/train_backdoor.py",
        "--trigger_size", str(trigger_size),
        "--target_label", str(TARGET_LABEL),
        "--poison_rate", str(POISON_RATE if not is_clean else 0.0),
        "--seed", str(seed),
        "--out_path", MODELS_DIR
    ]
    subprocess.run(train_cmd, check=True)

    # Extract Accuracy and ASR from the generated JSON
    metrics_path = os.path.join(MODELS_DIR, f"metrics_size{trigger_size}_seed{seed}.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    clean_acc = metrics["clean_accuracy"]
    asr = metrics["attack_success_rate"]

    # 2. Run Neural Cleanse Reverse Engineering + MAD Detection (Timed)
    # ---------------------------------------------------------
    detect_start = time.perf_counter()

    model_path = os.path.join(MODELS_DIR, f"model_size{trigger_size}_seed{seed}.h5")
    vis_out_dir = os.path.join(RESULTS_DIR, f"vis_size{trigger_size}_seed{seed}")

    vis_cmd = [
        "python", "mnist_visualize.py",
        "--model_path", model_path,
        "--out_dir", vis_out_dir
    ]
    subprocess.run(vis_cmd, check=True)

    mad_cmd = [
        "python", "mad_outlier_detection.py",
        "--input_dir", vis_out_dir
    ]
    result = subprocess.run(mad_cmd, capture_output=True, text=True, check=True)

    runtime_sec = round(time.perf_counter() - detect_start, 2)

    # 3. Parse MAD Output & Map Metrics
    # ---------------------------------------------------------
    anomaly_index, flagged_label = parse_mad_output(result.stdout)

    matches_true_target = (flagged_label == TARGET_LABEL) if not is_clean else (flagged_label == -1)

    # max_anomaly_index is the highest outlier index reported by MAD.
    # For infected models, attribute it to true_target_anomaly_index if the
    # true target was flagged (or if no label was flagged, record the score).
    max_anomaly_index = anomaly_index
    if is_clean:
        true_target_anomaly_index = 0.0
    elif flagged_label == TARGET_LABEL or flagged_label == -1:
        true_target_anomaly_index = anomaly_index
    else:
        # Wrong label was flagged instead of TARGET_LABEL
        true_target_anomaly_index = 0.0

    # 4. Save to CSV
    # ---------------------------------------------------------
    row = [
        DATASET, model_type, is_clean, trigger_size, seed, effective_target,
        clean_acc, asr,
        anomaly_index, true_target_anomaly_index, max_anomaly_index,
        flagged_label, matches_true_target, runtime_sec
    ]
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(
        f"Finished {model_type} size={trigger_size} seed={seed} "
        f"(anomaly={anomaly_index:.2f}, flagged={flagged_label}, time={runtime_sec}s). "
        f"Logged to {CSV_PATH}."
    )


if __name__ == "__main__":
    # Run Clean Models Control (Trigger Size 0)
    for seed in SEEDS:
        run_experiment(trigger_size=0, seed=seed, model_type="clean")

    # Run Backdoored Models Sweep
    for size in SIZES:
        for seed in SEEDS:
            run_experiment(trigger_size=size, seed=seed, model_type="backdoored")

    print("\nSweep fully completed. Results saved to", CSV_PATH)
