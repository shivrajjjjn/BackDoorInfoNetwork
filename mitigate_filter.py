#!/usr/bin/env python3
"""
Neural Cleanse Mitigation Check: Activation-Based Input Filtering (ECE 424 / M07)

For every correctly flagged backdoored model in results/sweep.csv:
  1. Loads the trained .h5 model and the reverse-engineered mask/pattern.
  2. Computes penultimate-layer neuron activations on clean vs. reverse-engineered
     triggered inputs to identify trigger-sensitive neurons.
  3. Calibrates an input-filter threshold on a clean validation split (target FRR).
  4. Evaluates Clean Accuracy and Attack Success Rate (ASR) before vs. after filtering.
  5. Exports results/analysis/mitigation_results.csv and mitigation_table.tex.

Usage:
  python mitigate_filter.py --sweep_csv results/sweep.csv --outdir results/analysis
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# Data & Trigger Helpers
# ---------------------------------------------------------------------------

def load_mnist_test():
    """Load and normalize MNIST test set to [0, 1] with shape (N, 28, 28, 1)."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


def apply_square_trigger(images: np.ndarray, trigger_size: int) -> np.ndarray:
    """
    Apply a bottom-right white square patch trigger of size (trigger_size x trigger_size).
    Adjust position/value here if mnist/train_backdoor.py places the patch elsewhere.
    """
    poisoned = images.copy()
    if trigger_size > 0:
        poisoned[:, -trigger_size:, -trigger_size:, :] = 1.0
    return poisoned


def load_reverse_engineered_trigger(vis_dir: str, label: int, img_shape=(28, 28, 1)):
    """
    Locate and load the reverse-engineered mask and pattern for `label` from vis_dir.
    Supports both .npy and .png outputs from mnist_visualize.py.
    """
    mask_candidates = (
        glob.glob(os.path.join(vis_dir, f"*mask*label_{label}.*"))
        + glob.glob(os.path.join(vis_dir, f"*mask*{label}.*"))
    )
    pattern_candidates = (
        glob.glob(os.path.join(vis_dir, f"*pattern*label_{label}.*"))
        + glob.glob(os.path.join(vis_dir, f"*pattern*{label}.*"))
    )

    if not mask_candidates or not pattern_candidates:
        raise FileNotFoundError(
            f"Could not find mask/pattern for label {label} in {vis_dir}."
        )

    def _load_arr(path):
        if path.endswith(".npy"):
            arr = np.load(path).astype("float32")
        else:
            img = Image.open(path).convert("L" if img_shape[-1] == 1 else "RGB")
            arr = np.array(img, dtype="float32") / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    mask = _load_arr(mask_candidates[0])
    pattern = _load_arr(pattern_candidates[0])
    return mask, pattern


def apply_reversed_trigger(images: np.ndarray, mask: np.ndarray, pattern: np.ndarray):
    """Stamp reverse-engineered (mask, pattern) onto images."""
    return (1.0 - mask) * images + mask * pattern


# ---------------------------------------------------------------------------
# Activation-Based Input Filter (Neural Cleanse Mitigation)
# ---------------------------------------------------------------------------

def get_penultimate_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a sub-model outputting the penultimate layer's activations."""
    penultimate_layer = model.layers[-2]
    return tf.keras.Model(inputs=model.input, outputs=penultimate_layer.output)


def evaluate_input_filter(
    model: tf.keras.Model,
    x_val: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    trigger_size: int,
    target_label: int,
    mask: np.ndarray,
    pattern: np.ndarray,
    target_frr: float = 0.05,
    top_k_frac: float = 0.10,
):
    """
    Calibrate an activation-based input filter on x_val using the reverse-engineered
    trigger, then evaluate clean accuracy and ASR on x_eval before vs. after filtering.
    """
    feat_model = get_penultimate_model(model)

    # 1. Identify trigger-sensitive neurons using validation clean vs. reversed-trigger inputs
    x_val_rev = apply_reversed_trigger(x_val, mask, pattern)
    act_clean_val = feat_model.predict(x_val, batch_size=256, verbose=0)
    act_rev_val = feat_model.predict(x_val_rev, batch_size=256, verbose=0)

    # Flatten if penultimate layer is spatial
    act_clean_val = act_clean_val.reshape(len(x_val), -1)
    act_rev_val = act_rev_val.reshape(len(x_val), -1)

    diff = np.mean(act_rev_val, axis=0) - np.mean(act_clean_val, axis=0)
    k = max(1, int(len(diff) * top_k_frac))
    top_neurons = np.argsort(diff)[-k:]

    # 2. Calibrate rejection threshold on clean validation data at (1 - target_frr) percentile
    val_scores = np.mean(act_clean_val[:, top_neurons], axis=1)
    threshold = float(np.quantile(val_scores, 1.0 - target_frr))

    # 3. Evaluate BEFORE mitigation
    clean_preds = np.argmax(model.predict(x_eval, batch_size=256, verbose=0), axis=1)
    clean_acc_before = float(np.mean(clean_preds == y_eval))

    # For ASR, evaluate on non-target test samples stamped with the true trigger
    non_target_idx = np.where(y_eval != target_label)[0]
    x_poison = apply_square_trigger(x_eval[non_target_idx], trigger_size)
    poison_preds = np.argmax(model.predict(x_poison, batch_size=256, verbose=0), axis=1)
    asr_before = float(np.mean(poison_preds == target_label))

    # 4. Evaluate AFTER mitigation (rejected inputs are blocked)
    act_clean_eval = feat_model.predict(x_eval, batch_size=256, verbose=0).reshape(len(x_eval), -1)
    clean_scores = np.mean(act_clean_eval[:, top_neurons], axis=1)
    clean_rejected = clean_scores > threshold
    # Rejected clean inputs count as misclassified (cannot be served)
    clean_correct_after = (clean_preds == y_eval) & (~clean_rejected)
    clean_acc_after = float(np.mean(clean_correct_after))
    frr_actual = float(np.mean(clean_rejected))

    act_poison_eval = feat_model.predict(x_poison, batch_size=256, verbose=0).reshape(len(x_poison), -1)
    poison_scores = np.mean(act_poison_eval[:, top_neurons], axis=1)
    poison_rejected = poison_scores > threshold
    # Attack succeeds only if NOT rejected AND classified as target_label
    asr_after = float(np.mean((poison_preds == target_label) & (~poison_rejected)))
    poison_block_rate = float(np.mean(poison_rejected))

    return {
        "clean_acc_before": clean_acc_before,
        "clean_acc_after": clean_acc_after,
        "clean_acc_delta": clean_acc_after - clean_acc_before,
        "asr_before": asr_before,
        "asr_after": asr_after,
        "asr_delta": asr_after - asr_before,
        "false_rejection_rate": frr_actual,
        "poison_block_rate": poison_block_rate,
        "filter_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# LaTeX Export & Main
# ---------------------------------------------------------------------------

def export_latex_table(res_df: pd.DataFrame, outdir: str):
    """Export grouped mitigation summary table for the ICLR report."""
    grouped = res_df.groupby(["dataset", "trigger_size"], as_index=False).agg(
        n_flagged=("seed", "count"),
        clean_before=("clean_acc_before", "mean"),
        clean_after=("clean_acc_after", "mean"),
        asr_before=("asr_before", "mean"),
        asr_after=("asr_after", "mean"),
    )

    tex_path = os.path.join(outdir, "mitigation_table.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Input-filtering mitigation on correctly flagged models (target FRR $= 5\%$).}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Trigger (px)} & \textbf{Flagged} & \textbf{Clean Acc (Before $\to$ After)} & \textbf{ASR (Before $\to$ After)} \\",
        r"\hline",
    ]
    for _, r in grouped.iterrows():
        lines.append(
            f"{r['dataset']} & {int(r['trigger_size'])} & {int(r['n_flagged'])} & "
            f"{r['clean_before']*100:.1f}\\% $\\to$ {r['clean_after']*100:.1f}\\% & "
            f"{r['asr_before']*100:.1f}\\% $\\to$ {r['asr_after']*100:.1f}\\% \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neural Cleanse Input Filtering Mitigation Check")
    parser.add_argument("--sweep_csv", type=str, default="results/sweep.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--outdir", type=str, default="results/analysis")
    parser.add_argument("--target_frr", type=float, default=0.05, help="Clean false rejection budget")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.sweep_csv)

    # Filter to correctly flagged infected models
    flagged = df[(df["is_clean"] == 0) & (df["matches_true_target"] == True)]
    if flagged.empty:
        print("[WARN] No correctly flagged backdoored models found in", args.sweep_csv)
        return

    x_test, y_test = load_mnist_test()
    # Split MNIST test set: 2,000 for filter calibration, 8,000 for evaluation
    x_val, x_eval, y_eval = x_test[:2000], x_test[2000:], y_test[2000:]

    rows = []
    for _, row in flagged.iterrows():
        size = int(row["trigger_size"])
        seed = int(row["seed"])
        target = int(row["target_label"])
        dataset = row["dataset"]

        model_path = os.path.join(args.results_dir, "models", f"model_size{size}_seed{seed}.h5")
        vis_dir = os.path.join(args.results_dir, f"vis_size{size}_seed{seed}")

        print(f"Mitigating {dataset} size={size} seed={seed} (target={target})...")
        model = tf.keras.models.load_model(model_path, compile=False)
        mask, pattern = load_reverse_engineered_trigger(vis_dir, target, img_shape=x_val.shape[1:])

        metrics = evaluate_input_filter(
            model=model,
            x_val=x_val,
            x_eval=x_eval,
            y_eval=y_eval,
            trigger_size=size,
            target_label=target,
            mask=mask,
            pattern=pattern,
            target_frr=args.target_frr,
        )
        rows.append({
            "dataset": dataset,
            "trigger_size": size,
            "seed": seed,
            "target_label": target,
            **metrics,
        })

    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "mitigation_results.csv")
    res_df.to_csv(out_csv, index=False)
    export_latex_table(res_df, args.outdir)

    print("\n=== Mitigation Check Summary ===")
    print(res_df[[
        "dataset", "trigger_size", "seed",
        "clean_acc_before", "clean_acc_after",
        "asr_before", "asr_after", "poison_block_rate"
    ]].to_string(index=False))
    print(f"\n[DONE] Saved {out_csv} and {os.path.join(args.outdir, 'mitigation_table.tex')}")


if __name__ == "__main__":
    main()