"""
Biological Age Gap Analysis (Clean & Portable)
==============================================
Analyzes MoE biological-age predictions and writes all figures under
results/figures/bioage/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

from plot_style import apply_thesis_style

# Configuration
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures" / "bioage"
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
BIOAGE_RESULTS = RESULTS_DIR / "bioage_predictions.csv"
OUTLIERS_FLAGGED = RESULTS_DIR / "bioage_outliers_flagged.csv"

ROUTE_PALETTE = {"neural": "#1f77b4", "non_neural": "#ff7f0e"}
SAVE_KW = dict(dpi=300, bbox_inches="tight", pad_inches=0.08)


def log(msg):
    print(f"[analyze_bioage] {msg}", flush=True)


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def _calc_metrics(df: pd.DataFrame):
    mae = mean_absolute_error(df["AGE_MID"], df["PRED_AGE"])
    r2 = r2_score(df["AGE_MID"], df["PRED_AGE"])
    r, p = stats.pearsonr(df["AGE_MID"], df["PRED_AGE"])
    return mae, r2, r, p


def analyze_gaps(meta_test):
    section("Stage 2: Analyzing Biological Age Gaps")

    meta_test = meta_test.copy()
    meta_test["AGE_GAP"] = meta_test["PRED_AGE"] - meta_test["AGE_MID"]
    summary = meta_test.groupby("ROUTE")["AGE_GAP"].agg(["mean", "std", "median", "count"]).round(3)

    log("\nAge Gap Summary by Tissue Group:")
    print(summary)

    return meta_test, summary


def plot_results(df, mae, r2, r, p):
    section("Stage 3: Generating Visualizations")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()
    sns.set_theme(style="whitegrid", font_scale=1.15)

    plot_df = df.copy()
    plot_df["Tissue group"] = plot_df["ROUTE"].map(
        {"neural": "Neural", "non_neural": "Non-neural"}
    )

    # 1. Calibration plot (chronological vs predicted) -> Thesis Figure 4.7
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        data=plot_df,
        x="AGE_MID",
        y="PRED_AGE",
        hue="Tissue group",
        palette={"Neural": ROUTE_PALETTE["neural"], "Non-neural": ROUTE_PALETTE["non_neural"]},
        alpha=0.6,
        edgecolor="w",
        s=70,
        ax=ax,
    )
    slope, intercept, _, _, _ = stats.linregress(plot_df["AGE_MID"], plot_df["PRED_AGE"])
    xs = pd.Series([plot_df["AGE_MID"].min(), plot_df["AGE_MID"].max()])
    ax.plot(xs, intercept + slope * xs, "k--", lw=2, label="Fit line")
    min_val = min(plot_df["AGE_MID"].min(), plot_df["PRED_AGE"].min())
    max_val = max(plot_df["AGE_MID"].max(), plot_df["PRED_AGE"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r:", lw=2, label="Y = X")
    ax.text(
        0.03,
        0.97,
        f"MAE = {mae:.2f}\nR² = {r2:.3f}\nPearson r = {r:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.set_title("BioAge Clock Calibration", fontweight="bold", pad=8)
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted Biological Age (Years)")
    ax.legend(title="Tissue group")
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "01_calibration_plot.png", **SAVE_KW)
    plt.close(fig)
    log("Saved: 01_calibration_plot.png")

    # 2. Gap distribution -> Thesis Figure 4.9
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(plot_df["AGE_GAP"], kde=True, color="skyblue", stat="density", ax=ax)
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_title("Distribution of Biological Age Gap (Age Acceleration)", fontweight="bold", pad=8)
    ax.set_xlabel("Age Gap (Predicted - Chronological) in Years")
    ax.set_ylabel("Density")
    fig.savefig(FIGURES_DIR / "02_gap_distribution.png", **SAVE_KW)
    plt.close(fig)
    log("Saved: 02_gap_distribution.png")

    # 3. Tissue group comparison -> Thesis Figure 4.8
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(
        data=plot_df,
        x="Tissue group",
        y="AGE_GAP",
        hue="Tissue group",
        palette={"Neural": ROUTE_PALETTE["neural"], "Non-neural": ROUTE_PALETTE["non_neural"]},
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="-", lw=1.5)
    ax.set_title("Age Acceleration by Tissue Group", fontweight="bold", pad=8)
    ax.set_xlabel("Tissue Group")
    ax.set_ylabel("Age Gap (Years)")
    fig.savefig(FIGURES_DIR / "03_tissue_comparison.png", **SAVE_KW)
    plt.close(fig)
    log("Saved: 03_tissue_comparison.png")

    # 4. Age dependency (residuals vs chronological age) -> Thesis Figure 4.11
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x="AGE_MID",
        y="AGE_GAP",
        hue="Tissue group",
        palette={"Neural": ROUTE_PALETTE["neural"], "Non-neural": ROUTE_PALETTE["non_neural"]},
        alpha=0.6,
        edgecolor="w",
        s=70,
        ax=ax,
    )
    ax.axhline(0, color="red", linestyle="--", lw=2)
    ax.set_title("Age Dependency of the BioAge Clock", fontweight="bold", pad=8)
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Age Gap (Years)")
    ax.legend(title="Tissue group")
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "04_age_dependency.png", **SAVE_KW)
    plt.close(fig)
    log("Saved: 04_age_dependency.png")

    # 5. Absolute error by age decade -> Thesis Figure 4.10
    plot_df["Age decade"] = pd.cut(
        plot_df["AGE_MID"],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["20s", "30s", "40s", "50s", "60s", "70s"],
        right=False,
    )
    plot_df["Abs error"] = (plot_df["PRED_AGE"] - plot_df["AGE_MID"]).abs()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="Age decade", y="Abs error", color="#2ca02c", ax=ax)
    ax.axhline(mae, color="red", linestyle="--", lw=2, label=f"Overall MAE = {mae:.2f}")
    ax.set_title("Absolute Prediction Error by Age Decade", fontweight="bold", pad=8)
    ax.set_xlabel("Chronological age decade")
    ax.set_ylabel("Absolute error (years)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIGURES_DIR / "05_error_by_decade.png", **SAVE_KW)
    plt.close(fig)
    log("Saved: 05_error_by_decade.png")


def export_flagged_outliers(df: pd.DataFrame, mae: float):
    abs_err = (df["PRED_AGE"] - df["AGE_MID"]).abs()
    outliers = df.loc[abs_err > 2 * mae, ["SAMPID", "AGE_MID", "PRED_AGE"]].copy()
    outliers["ABS_ERR"] = abs_err.loc[outliers.index]
    outliers.to_csv(OUTLIERS_FLAGGED, index=False)
    log(f"Flagged {len(outliers)} outliers (>2x MAE) to {OUTLIERS_FLAGGED}")


def main():
    log("=" * 60)
    log("Biological Age Gap Analysis Pipeline (Clean & Portable)")
    log("=" * 60)

    try:
        if not PREDICTIONS_FILE.exists():
            log(f"CRITICAL: {PREDICTIONS_FILE} not found. Please run train_moe_en.py first.")
            return

        section("Loading Test Predictions")
        df_test = pd.read_csv(PREDICTIONS_FILE)
        log(f"Loaded {len(df_test)} test samples.")

        mae, r2, r, p = _calc_metrics(df_test)
        log(f"Metrics - MAE: {mae:.2f}, R2: {r2:.3f}, Pearson r: {r:.3f} (p={p:.1e})")

        df_analyzed, summary = analyze_gaps(df_test)
        df_analyzed.to_csv(BIOAGE_RESULTS, index=False)
        log(f"Analyzed results saved to {BIOAGE_RESULTS}")

        plot_results(df_analyzed, mae, r2, r, p)
        export_flagged_outliers(df_analyzed, mae)

        section("Analysis Complete")
        print("\nFinal BioAge Summary:")
        print(summary)
        log(f"All figures saved to: {FIGURES_DIR}")

    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    log("=" * 60)


if __name__ == "__main__":
    main()
