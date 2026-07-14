"""Helpers for applying published REG / Pasta transcriptomic clocks (pyaging spec)."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

RAW_EXPRESSION = "data/raw/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"

REG_INTERCEPT = 140.272578432562
PASTA_SCALE = -4.76348378687217
PASTA_OFFSET = -0.0502893445253186


def load_ensg_index():
    """Return array of stripped ENSG IDs aligned to parquet rows."""
    names = (
        pq.read_table(RAW_EXPRESSION, columns=["Name"])
        .column("Name")
        .to_pandas()
        .astype(str)
    )
    return names.str.replace(r"\.\d+$", "", regex=True).values


def load_coefficients(coeff_path):
    """Load clock coefficients in feature order."""
    df = pd.read_csv(coeff_path)
    gene_col, coef_col = df.columns[0], df.columns[1]
    genes = df[gene_col].astype(str).tolist()
    coefs = df[coef_col].astype(float).values
    return genes, coefs


def load_clock_expression(sample_ids, required_genes):
    """Load raw TPM for required ENSG genes and test samples (samples x genes)."""
    ensg_by_row = load_ensg_index()
    gene_set = set(required_genes)

    table = pq.read_table(RAW_EXPRESSION, columns=list(sample_ids))
    expr = table.to_pandas().astype(np.float32)
    expr.index = ensg_by_row
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.loc[expr.index.isin(gene_set)]

    missing = [g for g in required_genes if g not in expr.index]
    if missing:
        print(
            f"[clock_utils] Warning: {len(missing)} clock genes absent in GTEx "
            f"(filled with 0 TPM), e.g. {missing[:3]}",
            flush=True,
        )

    # Build full matrix in coefficient order; missing genes -> 0 TPM
    available = expr.reindex(required_genes).T
    available = available.fillna(0.0)
    available.index.name = "SAMPID"
    return available


def rank_normalize(expr):
    """Median-fill missing values, then per-sample rank normalization (1-based)."""
    values = expr.values.astype(np.float64)
    median = np.nanmedian(values)
    if np.isnan(median):
        median = 0.0
    values = np.where(np.isnan(values), median, values)

    ranked = np.empty_like(values)
    for i in range(values.shape[0]):
        ranked[i] = rankdata(values[i], method="average")
    return ranked


def predict_linear_clock(expr_genes, coefs, postprocess):
    """Apply rank normalization, linear score, and clock-specific post-processing."""
    ranked = rank_normalize(expr_genes)
    scores = ranked @ coefs
    return postprocess(scores)


def postprocess_reg(scores):
    return scores + REG_INTERCEPT


def postprocess_pasta(scores):
    return scores * PASTA_SCALE + PASTA_OFFSET * PASTA_SCALE
