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
    gene_list.txt                 - retained ENSG gene IDs (version stripped)
    gene_stats.csv                - mean, variance, expr_frac per gene
    gene_annotation.csv           - ENSG, gene_symbol, name_versioned + stats

Run from repo root:
    python src/preprocess.py
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
CHUNK_SIZE      = 500
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
        if (i+1) % 5 == 0 or i == n_chunks-1:
            log(f"  Chunk {i+1:>3}/{n_chunks} | samples: {total_n:,}")

    stats = pd.DataFrame({
        "mean":      (gene_sum / total_n).values,
        "variance":  ((gene_sum_sq / total_n) - (gene_sum/total_n)**2).values,
        "expr_frac": (gene_count / total_n).values
    }, index=gene_ids.values)
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
    """Write ENSG -> gene symbol mapping for selected genes."""
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
    log(f"Examples: {out.iloc[0]['ENSG']} ({out.iloc[0]['gene_symbol']}), "
        f"{out.iloc[1]['ENSG']} ({out.iloc[1]['gene_symbol']})")
    return out_path


def build_filtered_matrix(valid_cols, selected_genes, gene_ids, output_path):
    section("Stage 5: Pass 2 - Building Filtered Matrix")
    log(f"Building {len(valid_cols):,} samples x {len(selected_genes):,} genes...")

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


def validate_gene_annotation(annotation_path, selected_genes):
    annot = pd.read_csv(annotation_path)
    assert len(annot) == len(selected_genes), (
        f"Expected {len(selected_genes)} rows in gene_annotation.csv, got {len(annot)}"
    )
    for col in ("ENSG", "gene_symbol", "name_versioned", "mean", "variance", "expr_frac"):
        assert col in annot.columns, f"Missing column in gene_annotation.csv: {col}"
    bad = [g for g in annot["ENSG"] if not ENSG_PATTERN.match(str(g))]
    assert not bad, f"gene_annotation.csv has non-ENSG entries, e.g. {bad[:3]}"
    assert set(annot["ENSG"]) == set(selected_genes), (
        "gene_annotation.csv ENSG set differs from gene_list.txt"
    )


def validate_outputs(expr_path, meta_path, gene_list_path, annotation_path=None):
    section("Stage 7: Validating Outputs")

    expr = pd.read_parquet(expr_path)
    meta = pd.read_parquet(meta_path)
    genes = Path(gene_list_path).read_text().strip().split("\n")

    assert expr.shape[1] == TOP_VAR_GENES, (
        f"Expected {TOP_VAR_GENES} gene columns, got {expr.shape[1]}"
    )
    assert len(genes) == TOP_VAR_GENES, (
        f"Expected {TOP_VAR_GENES} genes in gene_list.txt, got {len(genes)}"
    )

    bad_genes = [g for g in genes if not ENSG_PATTERN.match(g)]
    assert not bad_genes, (
        f"gene_list.txt has non-ENSG entries, e.g. {bad_genes[:3]}"
    )

    assert expr.index.equals(meta.index), "Expression and metadata sample indices differ"
    for col in ("AGE_MID", "SMTSD", "SUBJID"):
        assert col in meta.columns, f"Missing metadata column: {col}"
    assert not meta["AGE_MID"].isna().any(), "AGE_MID contains NaN values"

    log(f"Expression shape : {expr.shape[0]:,} samples x {expr.shape[1]:,} genes")
    log(f"Metadata shape   : {meta.shape}")
    log(f"Gene ID example  : {genes[0]}")
    log(f"Sample alignment : OK")
    log(f"ENSG validation  : OK ({TOP_VAR_GENES:,} genes)")

    if annotation_path is not None:
        validate_gene_annotation(annotation_path, genes)
        log(f"Gene annotation : OK ({TOP_VAR_GENES:,} genes)")


def build_annotation_only():
    """Regenerate gene_annotation.csv from existing gene_list.txt and gene_stats.csv."""
    gene_list_path = OUTPUT_DIR / "gene_list.txt"
    stats_path = OUTPUT_DIR / "gene_stats.csv"
    for path in (gene_list_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run full preprocess first.")
    selected_genes = gene_list_path.read_text().strip().split("\n")
    stats_df = pd.read_csv(stats_path, index_col=0)
    check_raw_files()
    annotation_path = save_gene_annotation(selected_genes, stats_df)
    validate_gene_annotation(annotation_path, selected_genes)
    log(f"Annotation-only complete -> {annotation_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GTEx v11 preprocessing pipeline")
    parser.add_argument(
        "--annotation-only",
        action="store_true",
        help="Regenerate gene_annotation.csv from existing processed outputs",
    )
    args = parser.parse_args()

    log("=" * 60)
    log("GTEx v11 Preprocessing Pipeline")
    log("=" * 60)

    if args.annotation_only:
        build_annotation_only()
        log("=" * 60)
        return

    check_raw_files()
    meta           = load_metadata()
    valid_samples  = set(meta.index.tolist())
    valid_cols     = get_sample_columns(valid_samples)
    gene_ids       = load_gene_ids()
    if len(gene_ids) != pq.read_metadata(EXPRESSION_FILE).num_rows:
        raise ValueError("Gene ID count does not match expression matrix row count")
    gene_stats     = compute_gene_stats(valid_cols, gene_ids)
    selected_genes, stats_qc = select_genes(gene_stats)
    annotation_path = save_gene_annotation(selected_genes, stats_qc)

    gene_list_path = OUTPUT_DIR / "gene_list.txt"
    with open(gene_list_path, "w") as f:
        f.write("\n".join(selected_genes))
    log(f"Gene list saved -> {gene_list_path}")

    expr_out = OUTPUT_DIR / "expression_filtered.parquet"
    build_filtered_matrix(valid_cols, selected_genes, gene_ids, expr_out)

    meta_out = OUTPUT_DIR / "metadata_filtered.parquet"
    save_metadata(meta, expr_out, meta_out)

    validate_outputs(expr_out, meta_out, gene_list_path, annotation_path)

    section("Preprocessing Complete")
    log(f"  Expression : {expr_out}")
    log(f"  Metadata   : {meta_out}")
    log(f"  Gene list  : {gene_list_path}")
    log(f"  Gene stats : {OUTPUT_DIR / 'gene_stats.csv'}")
    log(f"  Annotation : {annotation_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
