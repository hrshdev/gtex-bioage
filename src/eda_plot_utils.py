"""Shared colours and helpers for GTEx preprocessed EDA figures."""

NEURAL_KEYWORDS = ("brain", "spinal cord", "nerve")

# High-contrast pair (not blue vs purple)
DOMAIN_COLORS = {
    "neural": "#D95F02",       # orange
    "non_neural": "#1B9E77",   # teal green
}
OTHER_COLOR = "#7A7A7A"


def is_neural_tissue(tissue: str) -> bool:
    if tissue is None:
        return False
    name = str(tissue).lower()
    return any(kw in name for kw in NEURAL_KEYWORDS)


def tissue_domain_color(tissue: str) -> str:
    if str(tissue).startswith("Other"):
        return OTHER_COLOR
    return DOMAIN_COLORS["neural"] if is_neural_tissue(tissue) else DOMAIN_COLORS["non_neural"]


def domain_legend_patches():
    import matplotlib.patches as mpatches

    return [
        mpatches.Patch(color=DOMAIN_COLORS["neural"], label="Neural tissues"),
        mpatches.Patch(color=DOMAIN_COLORS["non_neural"], label="Non-neural tissues"),
        mpatches.Patch(color=OTHER_COLOR, label="Other tissues"),
    ]
