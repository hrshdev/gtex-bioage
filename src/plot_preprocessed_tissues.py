"""
Generate full and top-N tissue distribution figures for preprocessed GTEx v11 EDA.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eda_plot_utils import (  # noqa: E402
    domain_legend_patches,
    is_neural_tissue,
    tissue_domain_color,
)
from plot_style import apply_thesis_style  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "results" / "figures" / "preprocessed"
RESULTS_DIR = REPO_ROOT / "results"

BG = "#F8F9FA"
TOP_N = 20
THESIS_TOP_N = 15


def load_metadata():
    meta_path = PROCESSED_DIR / "metadata_filtered.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run preprocess.py first.")
    return pq.read_table(meta_path).to_pandas()


def save_full_tissue_chart(meta: pd.DataFrame):
    tissue_counts = meta["SMTSD"].value_counts().sort_values(ascending=True)
    colors = [tissue_domain_color(t) for t in tissue_counts.index]

    fig, ax = plt.subplots(
        figsize=(12, max(12, len(tissue_counts) * 0.28)),
        facecolor=BG,
    )
    fig.suptitle(
        "GTEx v11 Processed — All Tissues Sample Distribution",
        fontsize=14,
        fontweight="bold",
    )

    bars = ax.barh(
        range(len(tissue_counts)),
        tissue_counts.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.75,
    )
    ax.set_yticks(range(len(tissue_counts)))
    ax.set_yticklabels(tissue_counts.index, fontsize=7.5)
    ax.set_xlabel("Number of Samples")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    for bar in bars:
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}",
            va="center",
            fontsize=6.5,
        )

    ax.legend(handles=domain_legend_patches()[:2], loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "02_tissue_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out.relative_to(REPO_ROOT)}")


def build_top_summary(tissue_counts: pd.Series) -> pd.Series:
    top = tissue_counts.head(TOP_N)
    other_n = int(tissue_counts.iloc[TOP_N:].sum())
    other_label = f"Other ({len(tissue_counts) - TOP_N} tissues)"
    return pd.concat([top, pd.Series({other_label: other_n})]).sort_values(ascending=True)


def save_top_tissue_chart(meta: pd.DataFrame):
    tissue_counts = meta["SMTSD"].value_counts()
    summary = build_top_summary(tissue_counts)
    colors = [tissue_domain_color(t) for t in summary.index]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    fig.suptitle(
        f"GTEx v11 Processed — Top {TOP_N} Tissues + Other",
        fontsize=14,
        fontweight="bold",
    )

    bars = ax.barh(
        range(len(summary)),
        summary.values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72,
    )
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index, fontsize=10)
    ax.set_xlabel("Number of Samples")
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    xmax = summary.max()
    ax.set_xlim(0, xmax * 1.10)

    for bar in bars:
        val = int(bar.get_width())
        label = f"{val:,}"
        y = bar.get_y() + bar.get_height() / 2
        if bar.get_width() >= xmax * 0.20:
            ax.text(
                bar.get_width() - xmax * 0.015,
                y,
                label,
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_width() + xmax * 0.012,
                y,
                label,
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.legend(handles=domain_legend_patches(), loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "02b_top_tissues_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out.relative_to(REPO_ROOT)}")


def save_thesis_tissue_summary(meta: pd.DataFrame):
    """Compact top-N + Other chart for thesis (readable, fits one page)."""
    apply_thesis_style()
    tissue_counts = meta["SMTSD"].value_counts()
    n_tissues = len(tissue_counts)
    top = tissue_counts.head(THESIS_TOP_N)
    other_n = int(tissue_counts.iloc[THESIS_TOP_N:].sum())
    other_label = f"Other ({n_tissues - THESIS_TOP_N} tissues)"
    summary = pd.concat([top, pd.Series({other_label: other_n})]).sort_values(
        ascending=True
    )
    colors = [tissue_domain_color(t) for t in summary.index]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    bars = ax.barh(
        range(len(summary)),
        summary.values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72,
    )
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Number of Samples")
    ax.set_title(
        f"GTEx v11 Processed — Top {THESIS_TOP_N} Tissues + Other ($n={len(meta):,}$)",
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    xmax = summary.max()
    ax.set_xlim(0, xmax * 1.12)
    for bar in bars:
        val = int(bar.get_width())
        label = f"{val:,}"
        y = bar.get_y() + bar.get_height() / 2
        if bar.get_width() >= xmax * 0.18:
            ax.text(
                bar.get_width() - xmax * 0.015,
                y,
                label,
                va="center",
                ha="right",
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(bar.get_width() + xmax * 0.012, y, label, va="center", ha="left")

    ax.legend(handles=domain_legend_patches(), loc="lower right")
    plt.tight_layout()

    out = FIGURES_DIR / "02c_thesis_tissue_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved -> {out.relative_to(REPO_ROOT)}")


def save_tissue_table(meta: pd.DataFrame):
    counts = meta["SMTSD"].value_counts().reset_index()
    counts.columns = ["tissue", "sample_count"]
    counts["domain"] = counts["tissue"].apply(
        lambda t: "neural" if is_neural_tissue(t) else "non_neural"
    )
    out = RESULTS_DIR / "tissue_sample_counts.csv"
    counts.to_csv(out, index=False)
    print(f"Saved -> {out.relative_to(REPO_ROOT)} ({len(counts)} tissues)")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    save_full_tissue_chart(meta)
    save_top_tissue_chart(meta)
    save_thesis_tissue_summary(meta)
    save_tissue_table(meta)


if __name__ == "__main__":
    main()
