"""
Comparative BioAge Analysis
===========================
Compares the custom MoE clock against published human transcriptomic clocks
(REG and Pasta) on the same GTEx test set.

Preprocessing follows pyaging / biolearn: raw TPM, median fill, per-sample
rank normalization, then linear coefficients with clock-specific post-processing.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from clock_utils import (
    load_clock_expression,
    load_coefficients,
    postprocess_pasta,
    postprocess_reg,
    predict_linear_clock,
)
from plot_style import apply_thesis_style

RESULTS_DIR = Path("results")
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
REG_COEFFS = RESULTS_DIR / "REG_coeffs.csv"
PASTA_COEFFS = RESULTS_DIR / "Pasta_coeffs.csv"
COMPARISON_FILE = RESULTS_DIR / "clock_comparison.csv"
SUMMARY_FILE = RESULTS_DIR / "clock_benchmark_summary.csv"
COMPARISON_FIG = RESULTS_DIR / "figures" / "clocks" / "clock_comparison.png"


def log(msg):
    print(f"[compare_clocks] {msg}", flush=True)


def section(title):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'=' * 60}", flush=True)


def evaluate_clock(y_true, y_pred, model_name, notes=""):
    valid = ~pd.isna(y_pred)
    if not valid.any():
        return None
    yt = y_true[valid]
    yp = y_pred[valid]
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    r, p = pearsonr(yt, yp)
    return {
        "model": model_name,
        "mae": mae,
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p,
        "n_samples": int(valid.sum()),
        "notes": notes,
    }


def apply_clock(sample_ids, coeff_path, clock_name, postprocess):
    genes, coefs = load_coefficients(coeff_path)
    expr = load_clock_expression(sample_ids, genes)
    log(f"  {clock_name}: {expr.shape[0]:,} samples x {expr.shape[1]:,} genes")
    preds = predict_linear_clock(expr, coefs, postprocess)
    return pd.Series(preds, index=expr.index, name=f"{clock_name}_PRED")


def main():
    log("=" * 60)
    log("External Clock Benchmark (REG, Pasta)")
    log("=" * 60)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_FILE}. Run train_moe_en.py first.")

    df_test = pd.read_csv(PREDICTIONS_FILE)
    test_samples = df_test["SAMPID"].values
    log(f"Test samples: {len(test_samples):,}")

    section("Applying external clocks")
    reg_preds = apply_clock(test_samples, REG_COEFFS, "REG", postprocess_reg)
    pasta_preds = apply_clock(test_samples, PASTA_COEFFS, "Pasta", postprocess_pasta)

    comparison_df = df_test[["SAMPID", "AGE_MID", "PRED_AGE", "SMTSD", "ROUTE"]].copy()
    comparison_df.rename(columns={"PRED_AGE": "MoE_PRED"}, inplace=True)
    comparison_df["REG_PRED"] = reg_preds.reindex(test_samples).values
    comparison_df["PASTA_PRED"] = pasta_preds.reindex(test_samples).values

    section("Benchmark metrics")
    y_true = comparison_df["AGE_MID"]
    rows = [
        evaluate_clock(y_true, comparison_df["MoE_PRED"], "MoE_EN", "log1p TPM, top-5000 genes"),
        evaluate_clock(
            y_true, comparison_df["REG_PRED"], "REG",
            "rank-normalized TPM; age in years (Salignon et al. 2025)",
        ),
        evaluate_clock(
            y_true, comparison_df["PASTA_PRED"], "Pasta",
            "rank-normalized TPM; age-shift score rescaled (interpret Pearson cautiously)",
        ),
    ]
    summary_df = pd.DataFrame([r for r in rows if r is not None])
    print(summary_df[["model", "mae", "r2", "pearson_r", "notes"]].to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    COMPARISON_FIG.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    log(f"Saved -> {SUMMARY_FILE}")
    log(f"Saved -> {COMPARISON_FILE}")

    section("Generating comparison plot")
    plot_df = summary_df[summary_df["model"].isin(["MoE_EN", "REG", "Pasta"])].copy()
    label_map = {"MoE_EN": "MoE Elastic Net", "REG": "REG", "Pasta": "Pasta"}
    plot_df["model_label"] = plot_df["model"].map(label_map)

    apply_thesis_style()
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.15)
    ax = sns.barplot(
        data=plot_df,
        x="model_label",
        y="mae",
        hue="model_label",
        palette="viridis",
        legend=False,
    )
    ax.set_title("MAE Comparison: MoE vs External Human Clocks (GTEx test set)", fontweight="bold", pad=12)
    ax.set_ylabel("Mean Absolute Error (years)")
    ax.set_xlabel("Model")
    ymax = plot_df["mae"].max()
    ax.set_ylim(0, ymax * 1.15)
    for i, v in enumerate(plot_df["mae"]):
        ax.text(i, v + ymax * 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=15)
    plt.savefig(COMPARISON_FIG, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved -> {COMPARISON_FIG}")
    log("=" * 60)


if __name__ == "__main__":
    main()
