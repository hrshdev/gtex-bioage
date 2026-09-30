"""
GTEx v11 Biological Age Estimation - Data Preprocessing Pipeline (12.7 GB RAM Scaled)
====================================================================================
Scaled for 12.7 GB RAM environments with a 4.3 GB parquet file.
Uses CHUNK_SIZE = 1000 for 50% fewer I/O passes and faster execution.
"""

import gc
import re
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
CHUNK_SIZE      = 1000  # Increased from 500 to 1000 to maximize 12.7 GB RAM speed
MIN_EXPR        = 0.1
MIN_SAMPLE_FRAC = 0.10
TOP_VAR_GENES   = 5000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ENSG_PATTERN = re.compile(r"^ENSG\d+$")


def log(msg):
    print(f"[preprocess] {msg}", flush=True)


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def check_raw_files():
    """Fail early if required GTEx raw files are missing."""
    missing = [p for p in (EXPRESSION_FILE, SAMPLE_ATTR, SUBJECT_PHENO) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw data files:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nSee data/README.md for download instructions."
        )


def load_gene_id_table():
    """Load Name (versioned ENSG) and Description (gene symbol) from parquet."""
    table = pq.read_table(EXPRESSION_FILE, columns=["Name", "Description"])
    df = pd.DataFrame({
        "name_versioned": table.column("Name").to_pandas().astype(str),
        "gene_symbol": table.column("Description").to_pandas().astype(str),
    })
    del table
    gc.collect()
    df["ENSG"] = df["name_versioned"].str.replace(r"\.\d+$", "", regex=True)
    df = df[["ENSG", "gene_symbol", "name_versioned"]]
    n_dupes = df["ENSG"].duplicated().sum()
    if n_dupes:
        log(f"Warning: {n_dupes:,} duplicate ENSG IDs after version strip")
    return df


def load_gene_ids():
    """Load ENSG gene IDs from parquet Name column; strip Ensembl version suffix."""
    section("Stage 2b: Loading Gene IDs")
    id_table = load_gene_id_table()
    gene_ids = id_table["ENSG"]
    log(f"Loaded {len(gene_ids):,} gene IDs from Name column")
    log(f"Example IDs: {gene_ids.iloc[0]}, {gene_ids.iloc[1]}, {gene_ids.iloc[2]}")
    return gene_ids


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
    if "SEX" in meta.columns:
        meta["SEX"] = meta["SEX"].map({1: "M", 2: "F"})
    if "DTHHRDY" in meta.columns:
        before = len(meta)
        meta   = meta[meta["DTHHRDY"].isin([0,1,2,3,4])]
        log(f"After Hardy  : {len(meta):,} (dropped {before-len(meta)})")
    meta = meta.set_index("SAMPID")
    log(f"Final        : {len(meta):,} samples, {meta['SMTSD'].nunique()} tissues")
    del samples, subjects
    gc.collect()
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


def compute_gene_stats(valid_cols, gene_ids):
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
        log(f"  Chunk {i+1:>2}/{n_chunks} | samples: {total_n:,}")

    stats = pd.DataFrame({
        "mean":      (gene_sum / total_n).values,
        "variance":  ((gene_sum_sq / total_n) - (gene_sum/total_n)**2).values,
        "expr_frac": (gene_count / total_n).values
    }, index=gene_ids.values)
    del gene_sum, gene_sum_sq, gene_count
    gc.collect()
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
    if stats_qc.index.duplicated().any():
        before = len(stats_qc)
        stats_qc = stats_qc[~stats_qc.index.duplicated(keep="first")]
        log(f"Deduped ENSG IDs       : {before:,} -> {len(stats_qc):,}")
    if len(stats_qc) > TOP_VAR_GENES:
        stats_qc = stats_qc.nlargest(TOP_VAR_GENES, "variance")
    log(f"Final selected genes   : {len(stats_qc):,}")
    stats_qc.to_csv(OUTPUT_DIR / "gene_stats.csv")
    return stats_qc.index.tolist(), stats_qc


def save_gene_annotation(selected_genes, stats_df):
    section("Stage 4b: Saving Gene Annotation")
    annot = load_gene_id_table().drop_duplicates(subset="ENSG", keep="first")
    annot = annot.set_index("ENSG")

    missing = [g for g in selected_genes if g not in annot.index]
    if missing:
        raise ValueError(f"Missing annotation for {len(missing)} genes, e.g. {missing[:3]}")

    out = annot.loc[selected_genes].join(stats_df, how="left")
    out = out.reset_index()
    out_path = OUTPUT_DIR / "gene_annotation.csv"
    out.to_csv(out_path, index=False)
    log(f"Saved {len(out):,} gene annotations -> {out_path}")
    del annot, out
    gc.collect()
    return out_path


def build_filtered_matrix(valid_cols, selected_genes, gene_ids, output_path):
    section("Stage 5: Pass 2 - Building Filtered Matrix")
    log(f"Building {len(valid_cols):,} samples x {len(selected_genes):,} genes in chunks of {CHUNK_SIZE}...")

    selected_set = set(selected_genes)
    n_chunks     = (len(valid_cols) + CHUNK_SIZE - 1) // CHUNK_SIZE
    writer       = None
    total        = 0

    for i in range(n_chunks):
        cols = valid_cols[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE]
        df   = pq.read_table(EXPRESSION_FILE, columns=cols).to_pandas().astype(np.float32)
        df.index = gene_ids.values
        df = df[~df.index.duplicated(keep="first")]
        df = df.loc[df.index.isin(selected_set)].T
        df.index = pd.Index(cols, name="SAMPID")
        df = np.log1p(df)
        total += len(df)

        table = pa.Table.from_pandas(df.astype(np.float32), preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)
        del df, table
        gc.collect()
        log(f"  Chunk {i+1:>2}/{n_chunks} | written: {total:,}")

    if writer:
        writer.close()
    log(f"Done. {total:,} samples x {len(selected_genes):,} genes -> {output_path}")


def save_metadata(meta, expr_path, output_path):
    section("Stage 6: Saving Aligned Metadata")
    expr_ids = pd.read_parquet(str(expr_path), columns=[]).index.tolist()
    meta_aligned = meta[meta.index.isin(expr_ids)].copy()
    meta_aligned.to_parquet(str(output_path), compression="snappy")
    log(f"Saved {len(meta_aligned):,} samples -> {output_path}")
    del meta_aligned
    gc.collect()


def validate_outputs(expr_path, meta_path, gene_list_path, annotation_path=None):
    section("Stage 7: Validating Outputs")
    schema = pq.read_schema(str(expr_path))
    n_cols = len(schema.names)
    assert n_cols == TOP_VAR_GENES, f"Expected {TOP_VAR_GENES} genes, got {n_cols}"

    meta = pd.read_parquet(meta_path)
    genes = Path(gene_list_path).read_text().strip().split("\n")

    assert len(genes) == TOP_VAR_GENES, f"Expected {TOP_VAR_GENES} genes, got {len(genes)}"
    bad_genes = [g for g in genes if not ENSG_PATTERN.match(g)]
    assert not bad_genes, f"gene_list.txt has non-ENSG entries, e.g. {bad_genes[:3]}"

    for col in ("AGE_MID", "SMTSD", "SUBJID"):
        assert col in meta.columns, f"Missing metadata column: {col}"
    assert not meta["AGE_MID"].isna().any(), "AGE_MID contains NaN values"

    log(f"Verified Expression: ~{len(meta):,} samples x {n_cols:,} genes")
    log(f"Verified Metadata  : {meta.shape}")
    del meta
    gc.collect()


def main():
    log("=" * 60)
    log("GTEx v11 Preprocessing Pipeline (Scaled CHUNK_SIZE=1000)")
    log("=" * 60)

    check_raw_files()
    meta = load_metadata()
    valid_cols = get_sample_columns(meta.index.tolist())
    gene_ids = load_gene_ids()

    stats = compute_gene_stats(valid_cols, gene_ids)
    selected_genes, stats_qc = select_genes(stats)

    gene_list_path = OUTPUT_DIR / "gene_list.txt"
    gene_list_path.write_text("\n".join(selected_genes) + "\n")
    log(f"Saved gene list -> {gene_list_path}")

    annotation_path = save_gene_annotation(selected_genes, stats_qc)
    expr_path = OUTPUT_DIR / "expression_filtered.parquet"
    meta_path = OUTPUT_DIR / "metadata_filtered.parquet"

    build_filtered_matrix(valid_cols, selected_genes, gene_ids, expr_path)
    save_metadata(meta, expr_path, meta_path)
    validate_outputs(expr_path, meta_path, gene_list_path, annotation_path)

    log("\nPreprocessing completed at accelerated speed!")


if __name__ == "__main__":
    main()
