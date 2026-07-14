"""Shared helpers for loading and validating processed GTEx data."""

import re
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
EXPRESSION_FILE = PROCESSED_DIR / "expression_filtered.parquet"
METADATA_FILE = PROCESSED_DIR / "metadata_filtered.parquet"
GENE_LIST_FILE = PROCESSED_DIR / "gene_list.txt"

ENSG_PATTERN = re.compile(r"^ENSG\d+$")
REQUIRED_META_COLS = ("AGE_MID", "SMTSD", "SUBJID")


def check_processed_data():
    """Fail early if processed outputs are missing or invalid."""
    missing = [p for p in (EXPRESSION_FILE, METADATA_FILE, GENE_LIST_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed data not found:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nRun: python src/preprocess.py"
        )

    genes = GENE_LIST_FILE.read_text().strip().split("\n")
    if len(genes) != 5000:
        raise ValueError(f"Expected 5000 genes in gene_list.txt, got {len(genes)}")
    bad = [g for g in genes if not ENSG_PATTERN.match(g)]
    if bad:
        raise ValueError(f"Non-ENSG gene IDs in gene_list.txt: {bad[:3]}")


def load_processed_data():
    """Load aligned expression matrix and metadata."""
    check_processed_data()
    expr = pd.read_parquet(EXPRESSION_FILE)
    meta = pd.read_parquet(METADATA_FILE)

    common = expr.index.intersection(meta.index)
    expr = expr.loc[common]
    meta = meta.loc[common]

    if not expr.index.equals(meta.index):
        raise ValueError("Expression and metadata indices are not aligned")

    for col in REQUIRED_META_COLS:
        if col not in meta.columns:
            raise ValueError(f"Missing metadata column: {col}")

    return expr, meta


def make_age_bins(meta):
    """Age decade bins for stratified splitting (same as training scripts)."""
    return pd.cut(
        meta["AGE_MID"],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"],
        right=False,
    )
