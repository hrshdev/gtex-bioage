"""
Objective 3 — gene interpretation and pathway enrichment.

1. Merge MoE Elastic Net coefficients with gene symbols.
2. Export top clock genes per expert.
3. Run g:Profiler enrichment (KEGG + GO:BP) on top-weighted genes.
4. Save pathway table and bar-chart figure for presentation backup.
"""

import json
import math
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures" / "interpretation"
COEFF_FILE = RESULTS_DIR / "moe_gene_coefficients.csv"
ANNOTATION_FILE = Path("data/processed/gene_annotation.csv")
TOP_GENES_FILE = RESULTS_DIR / "top_clock_genes.csv"
PATHWAY_FILE = RESULTS_DIR / "pathway_enrichment.csv"
PATHWAY_FIG = FIGURES_DIR / "01_pathway_enrichment.png"

TOP_N_EXPORT = 50
TOP_N_ENRICH = 150
TOP_PATHWAYS_PLOT = 8
GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def log(msg):
    print(f"[interpret_genes] {msg}", flush=True)


def load_annotated_coefficients():
    coeffs = pd.read_csv(COEFF_FILE)
    annot = pd.read_csv(ANNOTATION_FILE)[["ENSG", "gene_symbol"]]
    merged = coeffs.merge(annot, on="ENSG", how="left")
    missing = merged["gene_symbol"].isna().sum()
    if missing:
        log(f"Warning: {missing} genes lack symbols — excluded from pathway enrichment")
    return merged


def export_top_genes(df):
    top = (
        df.sort_values(["expert", "abs_coefficient"], ascending=[True, False])
        .groupby("expert", group_keys=False)
        .head(TOP_N_EXPORT)
    )
    top.to_csv(TOP_GENES_FILE, index=False)
    log(f"Saved top {TOP_N_EXPORT} genes per expert -> {TOP_GENES_FILE}")
    for expert in top["expert"].unique():
        sub = top[top["expert"] == expert].head(5)
        symbols = ", ".join(sub["gene_symbol"].dropna().astype(str).tolist())
        log(f"  {expert}: {symbols}")
    return top


def _gprofiler_enrich(gene_symbols, expert_label):
    payload = {
        "organism": "hsapiens",
        "query": gene_symbols,
        "sources": ["KEGG", "GO:BP"],
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "no_evidences": False,
    }
    req = urllib.request.Request(
        GPROFILER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        log(f"g:Profiler request failed for {expert_label}: {exc}")
        return pd.DataFrame()

    rows = []
    for hit in data.get("result", []) or []:
        intersections = hit.get("intersections") or []
        gene_count = len(intersections[0]) if intersections else 0
        rows.append(
            {
                "expert": expert_label,
                "source": hit.get("source"),
                "term_id": hit.get("native"),
                "term_name": hit.get("name"),
                "p_value": hit.get("p_value"),
                "fdr": hit.get("adjusted_p_value") or hit.get("p_value"),
                "gene_count": gene_count,
            }
        )
    return pd.DataFrame(rows)


def run_pathway_enrichment(df):
    frames = []
    for expert in sorted(df["expert"].unique()):
        sub = df[df["expert"] == expert].sort_values("abs_coefficient", ascending=False)
        symbols = (
            sub["gene_symbol"]
            .dropna()
            .astype(str)
            .loc[lambda s: s.str.len() > 0]
            .head(TOP_N_ENRICH)
            .tolist()
        )
        if len(symbols) < 10:
            log(f"Skipping {expert}: too few symbols for enrichment")
            continue
        log(f"Enriching {expert}: {len(symbols)} genes via g:Profiler...")
        enriched = _gprofiler_enrich(symbols, expert)
        if enriched.empty:
            log(f"  No significant pathways for {expert}")
        else:
            log(f"  {len(enriched)} terms returned for {expert}")
        frames.append(enriched)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["p_value"] = pd.to_numeric(out["p_value"], errors="coerce")
    out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce").fillna(out["p_value"])
    out = out.sort_values(["expert", "p_value"])
    out.to_csv(PATHWAY_FILE, index=False)
    log(f"Saved pathway enrichment -> {PATHWAY_FILE}")
    return out


def _pathway_label(name: str, max_chars: int = 52) -> str:
    """Wrap long pathway names so labels are not truncated in print."""
    text = str(name).strip()
    if len(text) <= max_chars:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars))


def plot_pathways(pathways):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if pathways.empty:
        log("No pathways to plot — skipping figure")
        return

    plot_rows = []
    for expert in pathways["expert"].unique():
        sub = pathways[pathways["expert"] == expert].nsmallest(TOP_PATHWAYS_PLOT, "p_value").copy()
        sub["label"] = sub["term_name"].map(_pathway_label)
        sub["neg_log_p"] = -sub["p_value"].clip(lower=1e-300).map(math.log10)
        plot_rows.append(sub)

    plot_df = pd.concat(plot_rows, ignore_index=True)
    experts = list(plot_df["expert"].unique())

    # Stacked panels use full thesis width and leave room for long pathway names.
    fig, axes = plt.subplots(
        len(experts),
        1,
        figsize=(12, 4.2 * len(experts)),
        squeeze=False,
    )

    for ax, expert in zip(axes.flat, experts):
        sub = plot_df[plot_df["expert"] == expert].sort_values("neg_log_p")
        color = "#2E86AB" if expert == "neural" else "#E94F37"
        ax.barh(sub["label"], sub["neg_log_p"], color=color, height=0.72)
        ax.set_xlabel(r"$-\log_{10}$(p-value)", fontsize=13)
        ax.set_title(f"{expert.replace('_', ' ').title()} expert", fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()

    fig.suptitle(
        "Top enriched pathways (KEGG + GO:BP), MoE clock genes",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.38, right=0.97, top=0.93, bottom=0.08, hspace=0.45)
    fig.savefig(PATHWAY_FIG, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    log(f"Saved pathway figure -> {PATHWAY_FIG}")




def export_tissue_signatures():
    pred_file = RESULTS_DIR / "bioage_predictions.csv"
    if not pred_file.exists():
        log("Skipping tissue signatures — bioage_predictions.csv not found")
        return
    pred = pd.read_csv(pred_file)
    summary = pred.groupby("ROUTE")["AGE_GAP"].agg(["mean", "std", "count"]).round(3)
    out = RESULTS_DIR / "multi_tissue_signature_summary.csv"
    summary.to_csv(out)
    log(f"Saved tissue route signatures -> {out}")


def main():
    log("Objective 3 — gene interpretation")
    if not COEFF_FILE.exists():
        raise FileNotFoundError(f"Missing {COEFF_FILE} — run train_moe_en.py first")
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"Missing {ANNOTATION_FILE} — run preprocess.py first")

    merged = load_annotated_coefficients()
    export_top_genes(merged)
    pathways = run_pathway_enrichment(merged)
    plot_pathways(pathways)
    export_tissue_signatures()
    log("Done.")


if __name__ == "__main__":
    main()
