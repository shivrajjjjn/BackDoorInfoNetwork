#!/usr/bin/env python3
"""
Statistical Analysis & Threshold Validation for Neural Cleanse (ECE 424 / M07)
Owner: Roger Mendez (rtm141)

Covers:
  - Exp 1: Clean-model False Positive Rate (FPR) & MAD threshold sweep validation
  - Exp 2: Trigger-size failure boundary, Wilson 95% CIs, Fisher's Exact Test
           (with Holm-Bonferroni correction), and Mann-Whitney U tests
  - Outputs: Summary CSVs, matplotlib plots, and a LaTeX table for the report.

Usage:
  python analyze_results.py --input results/sweep_results.csv --outdir results/analysis
  python analyze_results.py --demo --outdir results/demo_analysis
"""

import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helper Statistics
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, confidence: float = 0.95):
    """Wilson score confidence interval (accurate for small n like 2 or 5 seeds)."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


# ---------------------------------------------------------------------------
# Experiment 1: Threshold Validation & Clean Controls
# ---------------------------------------------------------------------------

def run_threshold_validation(df: pd.DataFrame, thresholds: list) -> pd.DataFrame:
    """
    Evaluate TPR (Detection Rate on infected models) and FPR (on clean controls)
    across candidate MAD anomaly index thresholds.
    """
    rows = []
    for dataset, ddf in df.groupby("dataset"):
        clean = ddf[ddf["is_clean"] == 1]
        infected = ddf[ddf["is_clean"] == 0]

        n_clean = len(clean)
        n_inf = len(infected)

        for tau in thresholds:
            # Clean model is a false positive if ANY label exceeds threshold tau
            fp = int((clean["max_anomaly_index"] > tau).sum()) if n_clean > 0 else 0
            fpr = fp / n_clean if n_clean > 0 else np.nan
            fpr_lo, fpr_hi = wilson_ci(fp, n_clean)

            # Infected model is detected if the TRUE target label exceeds tau
            tp = int((infected["true_target_anomaly_index"] > tau).sum()) if n_inf > 0 else 0
            tpr = tp / n_inf if n_inf > 0 else np.nan
            tpr_lo, tpr_hi = wilson_ci(tp, n_inf)

            fn = n_inf - tp
            tn = n_clean - fp
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else np.nan

            rows.append({
                "dataset": dataset,
                "threshold": tau,
                "n_infected": n_inf,
                "TP": tp,
                "FN": fn,
                "TPR": tpr,
                "TPR_CI_low": tpr_lo,
                "TPR_CI_high": tpr_hi,
                "n_clean": n_clean,
                "FP": fp,
                "TN": tn,
                "FPR": fpr,
                "FPR_CI_low": fpr_lo,
                "FPR_CI_high": fpr_hi,
                "F1": f1,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 2: Trigger-Size Sweep & Statistical Significance
# ---------------------------------------------------------------------------

def run_trigger_size_summary(df: pd.DataFrame, default_tau: float = 2.0) -> pd.DataFrame:
    """
    Summarize anomaly index and detection rate by (dataset, trigger_size),
    including 95% Wilson CIs and Mann-Whitney U test against clean controls.
    """
    rows = []
    for dataset, ddf in df.groupby("dataset"):
        clean_vals = ddf[ddf["is_clean"] == 1]["max_anomaly_index"].values
        infected = ddf[ddf["is_clean"] == 0]

        for size, sdf in infected.groupby("trigger_size"):
            vals = sdf["true_target_anomaly_index"].values
            n = len(vals)
            detected = int((vals > default_tau).sum())
            det_rate = detected / n
            ci_lo, ci_hi = wilson_ci(detected, n)

            # Non-parametric test: Is infected anomaly index > clean control max index?
            if len(clean_vals) > 0 and n > 0:
                try:
                    u_stat, p_vs_clean = stats.mannwhitneyu(
                        vals, clean_vals, alternative="greater"
                    )
                except ValueError:
                    u_stat, p_vs_clean = np.nan, np.nan
            else:
                u_stat, p_vs_clean = np.nan, np.nan

            rows.append({
                "dataset": dataset,
                "trigger_size_px": size,
                "n_seeds": n,
                "mean_anomaly_idx": np.mean(vals),
                "std_anomaly_idx": np.std(vals, ddof=1) if n > 1 else 0.0,
                "median_anomaly_idx": np.median(vals),
                "detected_count": detected,
                "detection_rate": det_rate,
                "det_CI_low": ci_lo,
                "det_CI_high": ci_hi,
                "below_threshold_boundary": np.mean(vals) < default_tau,
                "mw_u_vs_clean": u_stat,
                "p_val_vs_clean": p_vs_clean,
            })

    return pd.DataFrame(rows)


def run_pairwise_significance(df: pd.DataFrame, default_tau: float = 2.0) -> pd.DataFrame:
    """
    Pairwise statistical tests across trigger sizes within each dataset:
      1. Fisher's Exact Test on binary detection rates (detected vs. missed)
      2. Mann-Whitney U Test on continuous anomaly indices
    Applies Holm-Bonferroni correction per dataset.
    """
    all_rows = []
    infected = df[df["is_clean"] == 0]

    for dataset, ddf in infected.groupby("dataset"):
        sizes = sorted(ddf["trigger_size"].unique())
        ds_rows = []

        for s1, s2 in combinations(sizes, 2):
            g1 = ddf[ddf["trigger_size"] == s1]["true_target_anomaly_index"].values
            g2 = ddf[ddf["trigger_size"] == s2]["true_target_anomaly_index"].values

            det1, miss1 = int((g1 > default_tau).sum()), int((g1 <= default_tau).sum())
            det2, miss2 = int((g2 > default_tau).sum()), int((g2 <= default_tau).sum())

            # 2x2 contingency table for Fisher's Exact Test
            table = [[det1, miss1], [det2, miss2]]
            odds_ratio, p_fisher = stats.fisher_exact(table, alternative="two-sided")

            # Mann-Whitney U test for continuous anomaly index shift
            try:
                _, p_mwu = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            except ValueError:
                p_mwu = np.nan

            ds_rows.append({
                "dataset": dataset,
                "size_A": s1,
                "size_B": s2,
                "det_rate_A": det1 / len(g1),
                "det_rate_B": det2 / len(g2),
                "fisher_odds_ratio": odds_ratio,
                "p_fisher_raw": p_fisher,
                "p_mwu_raw": p_mwu,
            })

        if ds_rows:
            raw_fisher = [r["p_fisher_raw"] for r in ds_rows]
            adj_fisher = holm_bonferroni(raw_fisher)
            for r, adj_p in zip(ds_rows, adj_fisher):
                r["p_fisher_holm"] = adj_p
                r["significant_alpha_0.05"] = adj_p < 0.05
            all_rows.extend(ds_rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Plotting & LaTeX Export
# ---------------------------------------------------------------------------

def plot_anomaly_vs_size(df: pd.DataFrame, summary_df: pd.DataFrame, outdir: str, tau: float = 2.0):
    """Plot mean anomaly index vs. trigger pixel area with clean-control band."""
    for dataset, sdf in summary_df.groupby("dataset"):
        clean = df[(df["dataset"] == dataset) & (df["is_clean"] == 1)]["max_anomaly_index"]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            sdf["trigger_size_px"],
            sdf["mean_anomaly_idx"],
            yerr=sdf["std_anomaly_idx"],
            fmt="-o",
            capsize=4,
            linewidth=2,
            label="Infected (True Target Anomaly Index)",
        )

        # Neural Cleanse threshold line
        ax.axhline(tau, color="red", linestyle="--", linewidth=1.5, label=f"MAD Threshold ({tau})")

        # Clean control reference band
        if len(clean) > 0:
            c_mean, c_std = clean.mean(), clean.std(ddof=1) if len(clean) > 1 else 0.0
            ax.axhspan(
                max(0, c_mean - c_std),
                c_mean + c_std,
                color="gray",
                alpha=0.2,
                label=f"Clean Control Max Index ({c_mean:.2f} ± {c_std:.2f})",
            )

        ax.set_title(f"{dataset}: Anomaly Index vs. Single-Patch Trigger Area")
        ax.set_xlabel("Trigger Area (pixels)")
        ax.set_ylabel("MAD Anomaly Index")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset.lower()}_anomaly_vs_size.png"), dpi=300)
        plt.close()


def plot_threshold_tradeoff(thresh_df: pd.DataFrame, outdir: str):
    """Plot TPR and FPR across MAD thresholds for each dataset."""
    for dataset, tdf in thresh_df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tdf["threshold"], tdf["TPR"], "-o", linewidth=2, label="Detection Rate (TPR)")
        ax.plot(tdf["threshold"], tdf["FPR"], "-s", linewidth=2, label="False Positive Rate (FPR)")
        ax.axvline(2.0, color="gray", linestyle="--", alpha=0.8, label="Default Cutoff (2.0)")

        ax.set_title(f"{dataset}: Detection vs. False Positive Rate by MAD Threshold")
        ax.set_xlabel("MAD Anomaly Index Threshold")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset.lower()}_threshold_tradeoff.png"), dpi=300)
        plt.close()


def export_latex_table(summary_df: pd.DataFrame, outdir: str):
    """Generate a clean LaTeX table ready for the ICLR report."""
    tex_path = os.path.join(outdir, "trigger_sweep_table.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Neural Cleanse detection performance across single-patch trigger sizes (MAD threshold $= 2.0$).}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Trigger (px)} & \textbf{Seeds} & \textbf{Anomaly Index (Mean $\pm$ SD)} & \textbf{Det. Rate} & \textbf{95\% Wilson CI} \\",
        r"\hline",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"{r['dataset']} & {int(r['trigger_size_px'])} & {int(r['n_seeds'])} & "
            f"{r['mean_anomaly_idx']:.2f} $\\pm$ {r['std_anomaly_idx']:.2f} & "
            f"{r['detection_rate']*100:.0f}\\% & "
            f"[{r['det_CI_low']*100:.0f}\\%, {r['det_CI_high']*100:.0f}\\%] \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Synthetic Demo Data Generator
# ---------------------------------------------------------------------------

def generate_demo_data() -> pd.DataFrame:
    """Creates realistic placeholder data matching the proposal's exact sweep design."""
    rng = np.random.default_rng(42)
    rows = []

    # MNIST: 6 sizes x 5 seeds + 5 clean controls
    mnist_specs = [(4, 3.8), (9, 3.4), (16, 2.7), (25, 2.1), (36, 1.6), (49, 1.2)]
    for size, mu in mnist_specs:
        for seed in range(1, 6):
            idx = max(0.2, rng.normal(mu, 0.35))
            rows.append({
                "dataset": "MNIST", "is_clean": 0, "trigger_size": size, "seed": seed,
                "target_label": 0, "true_target_anomaly_index": idx,
                "max_anomaly_index": max(idx, rng.uniform(1.1, 1.7)), "runtime_sec": rng.normal(120, 10)
            })
    for seed in range(1, 6):
        rows.append({
            "dataset": "MNIST", "is_clean": 1, "trigger_size": 0, "seed": seed,
            "target_label": -1, "true_target_anomaly_index": 0.0,
            "max_anomaly_index": rng.uniform(1.0, 2.1), "runtime_sec": rng.normal(115, 8)
        })

    # GTSRB: 3 sizes x 2 seeds + 2 clean controls
    gtsrb_specs = [(16, 3.5), (64, 2.3), (144, 1.4)]
    for size, mu in gtsrb_specs:
        for seed in range(1, 3):
            idx = max(0.2, rng.normal(mu, 0.25))
            rows.append({
                "dataset": "GTSRB", "is_clean": 0, "trigger_size": size, "seed": seed,
                "target_label": 33, "true_target_anomaly_index": idx,
                "max_anomaly_index": max(idx, rng.uniform(1.2, 1.8)), "runtime_sec": rng.normal(600, 30)
            })
    for seed in range(1, 3):
        rows.append({
            "dataset": "GTSRB", "is_clean": 1, "trigger_size": 0, "seed": seed,
            "target_label": -1, "true_target_anomaly_index": 0.0,
            "max_anomaly_index": rng.uniform(1.1, 1.85), "runtime_sec": rng.normal(590, 25)
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neural Cleanse Statistical & Threshold Analysis")
    parser.add_argument("--input", type=str, help="Path to sweep_results.csv")
    parser.add_argument("--outdir", type=str, default="results/analysis", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        df = generate_demo_data()
        df.to_csv(os.path.join(args.outdir, "demo_sweep_results.csv"), index=False)
        print("[INFO] Running in --demo mode with synthetic MNIST/GTSRB sweep data.\n")
    else:
        if not args.input or not os.path.exists(args.input):
            raise FileNotFoundError("Provide a valid --input CSV or pass --demo.")
        df = pd.read_csv(args.input)

    # 1. Runtime benchmark summary
    if "runtime_sec" in df.columns:
        print("=== Runtime Benchmark (per detection run) ===")
        rt = df.groupby("dataset")["runtime_sec"].agg(["count", "mean", "std", "sum"])
        rt["total_gpu_hours"] = rt["sum"] / 3600.0
        print(rt.to_string(), "\n")

    # 2. Exp 1: Threshold Validation & Clean Controls
    thresholds = [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    thresh_df = run_threshold_validation(df, thresholds)
    thresh_df.to_csv(os.path.join(args.outdir, "exp1_threshold_validation.csv"), index=False)
    print("=== Exp 1: Threshold Validation & Clean Controls ===")
    print(thresh_df[["dataset", "threshold", "TPR", "FPR", "F1"]].to_string(index=False), "\n")

    # 3. Exp 2: Trigger-Size Summary & Failure Boundary
    summary_df = run_trigger_size_summary(df, default_tau=2.0)
    summary_df.to_csv(os.path.join(args.outdir, "exp2_trigger_size_summary.csv"), index=False)
    print("=== Exp 2: Trigger-Size Sweep & Failure Boundary (tau = 2.0) ===")
    print(summary_df[[
        "dataset", "trigger_size_px", "mean_anomaly_idx", "detection_rate",
        "below_threshold_boundary", "p_val_vs_clean"
    ]].to_string(index=False), "\n")

    # 4. Exp 2: Pairwise Significance Tests
    sig_df = run_pairwise_significance(df, default_tau=2.0)
    sig_df.to_csv(os.path.join(args.outdir, "exp2_pairwise_significance.csv"), index=False)
    print("=== Exp 2: Pairwise Significance (Fisher's Exact + Holm-Bonferroni) ===")
    print(sig_df[[
        "dataset", "size_A", "size_B", "det_rate_A", "det_rate_B",
        "p_fisher_raw", "p_fisher_holm", "significant_alpha_0.05"
    ]].to_string(index=False), "\n")

    # 5. Plots & LaTeX table
    plot_anomaly_vs_size(df, summary_df, args.outdir, tau=2.0)
    plot_threshold_tradeoff(thresh_df, args.outdir)
    export_latex_table(summary_df, args.outdir)
    print(f"[DONE] Saved CSVs, plots, and LaTeX table to: {args.outdir}")


if __name__ == "__main__":
    main()