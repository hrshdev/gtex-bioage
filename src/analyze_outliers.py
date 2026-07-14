"""
BioAge Outlier Analysis
=======================
This script identifies samples with extreme Biological Age Gaps (Age Acceleration/Deceleration)
and analyzes their tissue distribution and age-related patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from plot_style import apply_thesis_style

# Configuration
RESULTS_DIR      = Path("results")
PREDICTIONS_FILE = RESULTS_DIR / "bioage_predictions.csv"
OUTLIER_FILE     = RESULTS_DIR / "bioage_outliers.csv"
FIGURES_DIR      = RESULTS_DIR / "figures" / "outliers"

def log(msg):
    print(f"[outlier_analysis] {msg}", flush=True)

def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)

def main():
    log("=" * 60)
    log("Biological Age Outlier Analysis")
    log("=" * 60)

    try:
        if not PREDICTIONS_FILE.exists():
            log(f"CRITICAL: {PREDICTIONS_FILE} not found.")
            return

        df = pd.read_csv(PREDICTIONS_FILE)
        log(f"Loaded {len(df)} samples.")

        # 1. Define Outliers
        # We'll use a threshold of +/- 10 years as "extreme" and 2*std as "statistical"
        std_gap = df['AGE_GAP'].std()
        threshold = 10.0
        log(f"Statistical threshold (2*std): {2*std_gap:.2f} years")
        log(f"Absolute threshold for extreme analysis: {threshold} years")

        extreme_older = df[df['AGE_GAP'] >= threshold].copy()
        extreme_younger = df[df['AGE_GAP'] <= -threshold].copy()

        log(f"Found {len(extreme_older)} extreme 'older' samples.")
        log(f"Found {len(extreme_younger)} extreme 'younger' samples.")

        # 2. Tissue Distribution of Outliers
        section("Tissue Distribution Analysis")

        def get_top_tissues(df_outliers, top_n=10):
            return df_outliers['SMTSD'].value_counts().head(top_n)

        older_tissues = get_top_tissues(extreme_older)
        younger_tissues = get_top_tissues(extreme_younger)

        print("\nTop 10 Tissues - Biologically OLDER:")
        print(older_tissues)
        print("\nTop 10 Tissues - Biologically YOUNGER:")
        print(younger_tissues)

        # 3. Save Outlier Lists
        outliers_all = pd.concat([extreme_older, extreme_younger]).sort_values('AGE_GAP', ascending=False)
        outliers_all.to_csv(OUTLIER_FILE, index=False)
        log(f"Outlier list saved to {OUTLIER_FILE}")

        # 4. Visualizations
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        apply_thesis_style()
        sns.set_theme(style="whitegrid", font_scale=1.15)

        plot_df = outliers_all.copy()
        plot_df["Tissue group"] = plot_df["ROUTE"].map(
            {"neural": "Neural", "non_neural": "Non-neural"}
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df,
            x="AGE_MID",
            y="AGE_GAP",
            hue="Tissue group",
            alpha=0.7,
            palette={"Neural": "#1f77b4", "Non-neural": "#ff7f0e"},
            s=80,
            ax=ax,
        )
        ax.axhline(0, color="black", lw=1.5)
        ax.axhline(threshold, color="red", linestyle="--", lw=2, label=f"+{threshold:.0f} y threshold")
        ax.axhline(-threshold, color="blue", linestyle="--", lw=2, label=f"-{threshold:.0f} y threshold")
        ax.set_title("Biological Age Gaps of Extreme Outliers", fontweight="bold", pad=12)
        ax.set_xlabel("Chronological Age (years)")
        ax.set_ylabel("Age Gap (Predicted - Chronological, years)")
        ax.legend(title="Tissue group")
        fig.savefig(FIGURES_DIR / "01_outlier_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log("Saved: 01_outlier_scatter.png")

        top_all_tissues = pd.concat([extreme_older, extreme_younger])["SMTSD"].value_counts().head(15).index
        df_top_tissues = df[df["SMTSD"].isin(top_all_tissues)].copy()

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.boxplot(
            data=df_top_tissues,
            x="AGE_GAP",
            y="SMTSD",
            palette="coolwarm",
            hue="SMTSD",
            legend=False,
            ax=ax,
        )
        ax.axvline(0, color="black", lw=1.5)
        ax.set_title("Age Gap Distribution for Most Volatile Tissues", fontweight="bold", pad=12)
        ax.set_xlabel("Age Gap (years)")
        ax.set_ylabel("Tissue")
        ax.tick_params(axis="y", labelsize=11)
        fig.savefig(FIGURES_DIR / "02_volatile_tissues.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log("Saved: 02_volatile_tissues.png")

        section("Outlier Analysis Complete")

    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    log("=" * 60)

if __name__ == "__main__":
    main()
