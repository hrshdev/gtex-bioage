"""Presentation-ready results figure for the Results slide (vertical frame)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = Path("results/figures/presentation")
OUT_FILE = OUT_DIR / "results_slide_mae.png"

# Slide frame: tall portrait box on left (~3:4 aspect ratio)
FIG_W_IN = 4.2
FIG_H_IN = 5.4
DPI = 220

# Presentation theme (matches navy + gold slides)
NAVY = "#1B3A6B"
GOLD = "#F2B705"
GREY = "#8A9BB0"
LIGHT_GREY = "#E8EDF3"
WHITE = "#FFFFFF"

MODELS = [
    ("MoE Elastic Net", 5.55, "proposed"),
    ("Elastic Net", 5.74, "baseline"),
    ("XGBoost", 7.33, "baseline"),
    ("Random Forest", 8.59, "baseline"),
    ("REG", 12.14, "external"),
]

COLORS = {
    "proposed": GOLD,
    "baseline": NAVY,
    "external": GREY,
}


def _style_axes(ax):
    ax.set_facecolor(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=NAVY, width=1.0)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.9, zorder=0)


def plot_results_slide():
    names = [m[0] for m in MODELS]
    maes = [m[1] for m in MODELS]
    groups = [m[2] for m in MODELS]
    colors = [COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor(WHITE)

    y = np.arange(len(names))
    bars = ax.barh(
        y,
        maes,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        height=0.58,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12, fontweight="medium", color=NAVY)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (years)", fontsize=14, fontweight="bold", color=NAVY, labelpad=10)
    ax.set_xlim(0, 14.5)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"], fontsize=11)
    _style_axes(ax)

    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{mae:.2f}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=NAVY,
        )

    legend_handles = [
        mpatches.Patch(facecolor=GOLD, edgecolor="white", label="Proposed"),
        mpatches.Patch(facecolor=NAVY, edgecolor="white", label="Baselines"),
        mpatches.Patch(facecolor=GREY, edgecolor="white", label="External"),
    ]

    fig.suptitle(
        "Model Performance\n(Sample-Stratified)",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        x=0.5,
        y=0.97,
        ha="center",
        va="top",
        linespacing=1.15,
    )

    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.11),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=9.5,
        title="Model type",
        title_fontsize=10,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    legend.get_title().set_ha("center")

    fig.subplots_adjust(left=0.34, right=0.95, top=0.82, bottom=0.28)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_FILE,
        dpi=DPI,
        facecolor=WHITE,
        pad_inches=0.18,
    )
    plt.close(fig)
    print(f"Saved -> {OUT_FILE} ({int(FIG_W_IN * DPI)} x {int(FIG_H_IN * DPI)} px)")


if __name__ == "__main__":
    plot_results_slide()
