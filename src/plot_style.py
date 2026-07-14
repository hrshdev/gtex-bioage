"""Matplotlib defaults for thesis-readable figures (full page width)."""

import matplotlib.pyplot as plt

THESIS_RC = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
}


def apply_thesis_style():
    plt.rcParams.update(THESIS_RC)
