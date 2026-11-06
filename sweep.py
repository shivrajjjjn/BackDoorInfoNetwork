# sweep.py
import os
import subprocess
import json
import csv

# Define the scaled-down protocol parameters
SIZES = [2, 4, 6]       # Small, medium, and intermediate triggers
SEEDS = [42, 100, 2026] # 3 random seeds
TARGET_LABEL = 0        # Fixed target label
POISON_RATE = 0.1       # Fixed 10% poison rate

RESULTS_DIR = "./results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
CSV_PATH = os.path.join(RESULTS_DIR, "sweep.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize CSV with the required columns
columns = [
    "model_type", "trigger_size", "seed", "clean_accuracy", 
    "attack_success_rate", "anomaly_index", "flagged_label", "matches_true_target"
]

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)

def run_experiment(trigger_size, seed, model_type):
    print(f"\n{'='*50}\nRunning {model_type} model: size={trigger_size}, seed={seed}\n{'='*50}")
    
    # 1. Train the Model
    # ---------------------------------------------------------
    train_cmd = [
        "python", "mnist/train_backdoor.py",
        "--trigger_size", str(trigger_size),
        "--target_label", str(TARGET_LABEL),
        "--poison_rate", str(POISON_RATE if trigger_size > 0 else 0.0),
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
    
    # 2. Run Neural Cleanse Reverse Engineering
    # ---------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, f"model_size{trigger_size}_seed{seed}.h5")
    vis_out_dir = os.path.join(RESULTS_DIR, f"vis_size{trigger_size}_seed{seed}")
    
    vis_cmd = [
        "python", "mnist_visualize.py",
        "--model_path", model_path,
        "--out_dir", vis_out_dir
    ]
    subprocess.run(vis_cmd, check=True)
    
    # 3. Run MAD Outlier Detection
    # ---------------------------------------------------------
    # Note: mad_outlier_detection.py usually takes the dict of norms and outputs the anomaly index.
    # Assuming mad_outlier_detection.py takes a directory and prints the final metrics to stdout:
    mad_cmd = [
        "python", "mad_outlier_detection.py",
        "--input_dir", vis_out_dir
    ]
    
    result = subprocess.run(mad_cmd, capture_output=True, text=True, check=True)
    output = result.stdout
    
    # Basic parsing of standard stdout for MAD script
    # (Adjust this parsing depending on how mad_outlier_detection.py formats its output)
    anomaly_index = 0.0
    flagged_label = -1
    
    for line in output.split('\n'):
        if "Anomaly index:" in line or "anomaly index:" in line.lower():
            try:
                anomaly_index = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Flagged label:" in line or "flagged label:" in line.lower():
            try:
                flagged_label = int(line.split(":")[-1].strip())
            except ValueError:
                pass

    matches_true_target = (flagged_label == TARGET_LABEL) if trigger_size > 0 else (flagged_label == -1)
    
    # 4. Save to CSV
    # ---------------------------------------------------------
    row = [
        model_type, trigger_size, seed, clean_acc, asr, 
        anomaly_index, flagged_label, matches_true_target
    ]
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
    print(f"Finished {model_type} size={trigger_size} seed={seed}. Logged to {CSV_PATH}.")

# Run Clean Models Control (Trigger Size 0)
for seed in SEEDS:
    run_experiment(trigger_size=0, seed=seed, model_type="clean")

# Run Backdoored Models Sweep
for size in SIZES:
    for seed in SEEDS:
        run_experiment(trigger_size=size, seed=seed, model_type="backdoored")

print("\nSweep fully completed. Results saved to", CSV_PATH)