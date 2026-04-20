"""
GTEx v11 Biological Age Estimation - Data Preprocessing Pipeline
================================================================
Designed for low-RAM environments (8GB RAM) with a 4.3GB parquet file.

Actual data structure (discovered via EDA):
    - Parquet: 74,628 genes (rows) x 19,790 columns
    - Columns: Description, [19,788 sample IDs], Name
    - Only 1 row group — cannot chunk by row groups
    - Strategy: chunk by selecting batches of sample COLUMNS

Files expected in data/raw/:
    GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet
    GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt
    GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt

Outputs in data/processed/:
    expression_filtered.parquet   - (samples x top 5000 genes), log1p normalised
    metadata_filtered.parquet     - aligned sample + subject metadata
    gene_list.txt                 - retained gene IDs
    gene_stats.csv                - mean, variance, expr_frac per gene

Run from repo root:
    python src/preprocess.py
"""

import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Configuration
RAW_DIR         = Path("data/raw")
OUTPUT_DIR      = Path("data/processed")
EXPRESSION_FILE = RAW_DIR / "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet"
SAMPLE_ATTR     = RAW_DIR / "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
SUBJECT_PHENO   = RAW_DIR / "GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt"
CHUNK_SIZE      = 500
MIN_EXPR        = 0.1
MIN_SAMPLE_FRAC = 0.10
TOP_VAR_GENES   = 5000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[preprocess] {msg}", flush=True)

def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def load_metadata():
    section("Stage 1: Loading Metadata")
    samples  = pd.read_csv(SAMPLE_ATTR,  sep="\t", low_memory=False)
    subjects = pd.read_csv(SUBJECT_PHENO, sep="\t", low_memory=False)
    samples  = samples.copy()
    samples["SUBJID"] = samples["SAMPID"].str.extract(r"(GTEX-[^-]+)")
    meta = samples.merge(subjects, on="SUBJID", how="inner")
    log(f"Raw samples  : {len(samples):,}")
    log(f"Merged       : {len(meta):,} samples")
    keep = [c for c in ["SAMPID","SUBJID","SMTSD","SMTS","AGE",
                         "SEX","DTHHRDY","SMRIN"] if c in meta.columns]
    meta = meta[keep].copy()
    required = [c for c in ["SMTSD","AGE","SEX","DTHHRDY"] if c in meta.columns]
    before   = len(meta)
    meta     = meta.dropna(subset=required)
    log(f"After dropna : {len(meta):,} (dropped {before-len(meta)})")
    try:
        meta["AGE_MID"] = meta["AGE"].str.extract(r"(\d+)").astype(float) + 5
    except Exception:
        meta["AGE_MID"] = meta["AGE"].astype(float)
    if "DTHHRDY" in meta.columns:
        before = len(meta)
        meta   = meta[meta["DTHHRDY"].isin([0,1,2,3,4])]
        log(f"After Hardy  : {len(meta):,} (dropped {before-len(meta)})")
    meta = meta.set_index("SAMPID")
    log(f"Final        : {len(meta):,} samples, {meta['SMTSD'].nunique()} tissues")
    return meta


def get_sample_columns(valid_samples):
    section("Stage 2: Reading Parquet Schema")
    schema    = pq.read_schema(EXPRESSION_FILE)
    all_cols  = schema.names
    skip_cols = {"Description", "Name"}
    sample_cols = [c for c in all_cols if c not in skip_cols]
    valid_cols  = [c for c in sample_cols if c in valid_samples]
    log(f"Parquet columns  : {len(all_cols):,}")
    log(f"Sample columns   : {len(sample_cols):,}")
    log(f"Matched to meta  : {len(valid_cols):,}")
    return valid_cols


def compute_gene_stats(valid_cols):
    section("Stage 3: Pass 1 - Computing Gene Statistics")
    log(f"Processing {len(valid_cols):,} samples in chunks of {CHUNK_SIZE}...")
    gene_sum = gene_sum_sq = gene_count = None
    total_n  = 0
    n_chunks = (len(valid_cols) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(n_chunks):
        cols = valid_cols[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
        df   = pq.read_table(EXPRESSION_FILE, columns=cols).to_pandas().astype(np.float32)
        df   = np.log1p(df)
        n    = df.shape[1]
        total_n += n
        if gene_sum is None:
            gene_sum    = df.sum(axis=1)
            gene_sum_sq = (df**2).sum(axis=1)
            gene_count  = (df>0).sum(axis=1)
        else:
            gene_sum    += df.sum(axis=1)
            gene_sum_sq += (df**2).sum(axis=1)
            gene_count  += (df>0).sum(axis=1)
        del df
        gc.collect()
        if (i+1) % 5 == 0 or i == n_chunks-1:
            log(f"  Chunk {i+1:>3}/{n_chunks} | samples: {total_n:,}")

    gene_index = pq.read_table(EXPRESSION_FILE, columns=[]).to_pandas().index.astype(str)
    stats = pd.DataFrame({
        "mean":      (gene_sum / total_n).values,
        "variance":  ((gene_sum_sq / total_n) - (gene_sum/total_n)**2).values,
        "expr_frac": (gene_count / total_n).values
    }, index=gene_index)
    log(f"Gene stats computed for {len(stats):,} genes")
    return stats


def select_genes(stats):
    section("Stage 4: Gene Selection")
    mask_expr = stats["mean"]      >= MIN_EXPR
    mask_frac = stats["expr_frac"] >= MIN_SAMPLE_FRAC
    stats_qc  = stats[mask_expr & mask_frac].copy()
    log(f"All genes              : {len(stats):,}")
    log(f"After mean filter      : {mask_expr.sum():,}")
    log(f"After expr-frac filter : {len(stats_qc):,}")
    if len(stats_qc) > TOP_VAR_GENES:
        stats_qc = stats_qc.nlargest(TOP_VAR_GENES, "variance")
    log(f"Final selected genes   : {len(stats_qc):,}")
    stats_qc.to_csv(OUTPUT_DIR / "gene_stats.csv")
    return stats_qc.index.tolist()


def build_filtered_matrix(valid_cols, selected_genes, output_path):
    section("Stage 5: Pass 2 - Building Filtered Matrix")
    log(f"Building {len(valid_cols):,} samples x {len(selected_genes):,} genes...")

    gene_index = pq.read_table(EXPRESSION_FILE, columns=[]).to_pandas().index.astype(str)
    n_chunks   = (len(valid_cols) + CHUNK_SIZE - 1) // CHUNK_SIZE
    writer     = None
    total      = 0

    for i in range(n_chunks):
        cols = valid_cols[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
        df   = pq.read_table(EXPRESSION_FILE, columns=cols).to_pandas().astype(np.float32)
        df.index = gene_index

        # Keep selected genes, transpose to samples x genes
        df = df.loc[df.index.isin(selected_genes)].T
        df.index = pd.Index(cols, name="SAMPID")
        df = np.log1p(df)
        total += len(df)

        table = pa.Table.from_pandas(df.astype(np.float32), preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema,
                                      compression="snappy")
        writer.write_table(table)
        del df, table
        gc.collect()

        if (i+1) % 5 == 0 or i == n_chunks-1:
            log(f"  Chunk {i+1:>3}/{n_chunks} | written: {total:,}")

    if writer:
        writer.close()
    log(f"Done. {total:,} samples x {len(selected_genes):,} genes -> {output_path}")


def save_metadata(meta, expr_path, output_path):
    section("Stage 6: Saving Aligned Metadata")
    expr_ids = pd.read_parquet(str(expr_path), columns=[]).index.tolist()
    meta_aligned = meta[meta.index.isin(expr_ids)].copy()
    meta_aligned.to_parquet(str(output_path), compression="snappy")
    log(f"Saved {len(meta_aligned):,} samples -> {output_path}")


def main():
    log("=" * 60)
    log("GTEx v11 Preprocessing Pipeline")
    log("=" * 60)

    meta           = load_metadata()
    valid_samples  = set(meta.index.tolist())
    valid_cols     = get_sample_columns(valid_samples)
    gene_stats     = compute_gene_stats(valid_cols)
    selected_genes = select_genes(gene_stats)

    gene_list_path = OUTPUT_DIR / "gene_list.txt"
    with open(gene_list_path, "w") as f:
        f.write("\n".join(selected_genes))
    log(f"Gene list saved -> {gene_list_path}")

    expr_out = OUTPUT_DIR / "expression_filtered.parquet"
    build_filtered_matrix(valid_cols, selected_genes, expr_out)

    meta_out = OUTPUT_DIR / "metadata_filtered.parquet"
    save_metadata(meta, expr_out, meta_out)

    section("Preprocessing Complete")
    log(f"  Expression : {expr_out}")
    log(f"  Metadata   : {meta_out}")
    log(f"  Gene list  : {gene_list_path}")
    log(f"  Gene stats : {OUTPUT_DIR / 'gene_stats.csv'}")
    log("=" * 60)


if __name__ == "__main__":
    main()