"""Generate five readable cohort overview figures (split from dashboard)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import load_processed_data
from eda_plot_utils import domain_legend_patches, tissue_domain_color
from plot_style import apply_thesis_style

PALETTE = {
    "primary": "#4C6EF5",
    "secondary": "#74C0FC",
    "accent": "#F76707",
    "green": "#51CF66",
    "pink": "#F06595",
    "teal": "#20C997",
    "gray": "#ADB5BD",
    "bg": "#F8F9FA",
    "dark": "#212529",
}

HARDY_MAP = {
    0: "Ventilator",
    1: "Violent/Fast",
    2: "Fast illness",
    3: "Intermediate",
    4: "Slow illness",
}

OUT_DIR = Path("results/figures/preprocessed/cohort_panels")
THESIS_DIR = Path(r"C:\Research\Research\Thesis\figures")

PANELS = [
    ("figure4.1.png", "sex"),
    ("figure4.2.png", "age"),
    ("figure4.3.png", "hardy"),
    ("figure4.4.png", "rin"),
]


def _age_counts(meta):
    if "AGE" in meta.columns:
        return meta["AGE"].value_counts().sort_index()
    bins = pd.cut(
        meta["AGE_MID"],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
    )
    return bins.value_counts().sort_index()


def plot_sex(meta, out_path):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE["bg"])
    sex_counts = meta["SEX"].value_counts()
    ax.pie(
        sex_counts,
        labels=sex_counts.index,
        colors=[PALETTE["primary"], PALETTE["pink"]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 16},
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    ax.set_title("Sex Distribution (Processed Cohort)", fontweight="bold", pad=16)
    ax.text(
        0,
        0,
        f"{sex_counts.sum():,}\nsamples",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_age(meta, out_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    age_counts = _age_counts(meta)
    colors_age = plt.cm.viridis(np.linspace(0.2, 0.9, len(age_counts)))
    bars = ax.bar(
        range(len(age_counts)),
        age_counts.values,
        color=colors_age,
        edgecolor="white",
        linewidth=1.5,
        width=0.7,
    )
    ax.set_xticks(range(len(age_counts)))
    ax.set_xticklabels(age_counts.index, rotation=30, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Age Group Distribution", fontweight="bold", pad=12)
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    ymax = max(age_counts.values)
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ymax * 0.01,
            f"{int(b.get_height()):,}",
            ha="center",
            fontsize=13,
        )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_hardy(meta, out_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    hc = meta["DTHHRDY"].map(HARDY_MAP).value_counts().sort_index()
    hcolors = [
        PALETTE["green"],
        PALETTE["teal"],
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["gray"],
    ]
    bars = ax.barh(
        range(len(hc)),
        hc.values,
        color=hcolors[: len(hc)],
        edgecolor="white",
        height=0.6,
    )
    ax.set_yticks(range(len(hc)))
    ax.set_yticklabels(hc.index)
    ax.set_xlabel("Samples")
    ax.set_title("Hardy Scale (Death Circumstances)", fontweight="bold", pad=12)
    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    xmax = max(hc.values)
    for b in bars:
        ax.text(
            b.get_width() + xmax * 0.01,
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width()):,}",
            va="center",
            fontsize=13,
        )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_tissues(meta, out_path):
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=PALETTE["bg"])
    tc = meta["SMTSD"].value_counts().head(20).sort_values(ascending=True)
    cols_t = [tissue_domain_color(t) for t in tc.index]
    bars = ax.barh(
        range(len(tc)),
        tc.values,
        color=cols_t,
        edgecolor="white",
        height=0.7,
    )
    ax.set_yticks(range(len(tc)))
    ax.set_yticklabels(tc.index)
    ax.set_xlabel("Samples")
    ax.set_title("Top 20 Tissues by Sample Count", fontweight="bold", pad=12)
    ax.legend(handles=domain_legend_patches(), loc="lower right")
    ax.xaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    xmax = max(tc.values)
    for b in bars:
        ax.text(
            b.get_width() + xmax * 0.01,
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width()):,}",
            va="center",
            fontsize=12,
        )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def plot_rin(meta, out_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=PALETTE["bg"])
    rin = meta["SMRIN"].dropna()
    ax.hist(rin, bins=30, color=PALETTE["teal"], edgecolor="white", alpha=0.9)
    ax.axvline(
        rin.median(),
        color=PALETTE["accent"],
        linestyle="--",
        linewidth=2,
        label=f"Median: {rin.median():.1f}",
    )
    ax.set_xlabel("RNA Integrity Number (RIN)")
    ax.set_ylabel("Samples")
    ax.set_title("RNA Quality (SMRIN)", fontweight="bold", pad=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--")
    ax.set_axisbelow(True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def main():
    apply_thesis_style()
    _, meta = load_processed_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    plotters = {
        "sex": plot_sex,
        "age": plot_age,
        "hardy": plot_hardy,
        "tissues": plot_tissues,
        "rin": plot_rin,
    }

    for filename, key in PANELS:
        out = OUT_DIR / filename
        plotters[key](meta, out)
        thesis_out = THESIS_DIR / filename
        thesis_out.write_bytes(out.read_bytes())
        print(f"Saved -> {out}")
        print(f"Copied -> {thesis_out}")


if __name__ == "__main__":
    main()
