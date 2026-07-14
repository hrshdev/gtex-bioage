"""
Subject-grouped internal validation.

Trains and evaluates MoE + baselines with a SUBJID split so no donor appears
in both train and test. Compares against existing sample-stratified results.
"""

import argparse
from pathlib import Path

import pandas as pd

from data_utils import load_processed_data
from split_utils import SPLIT_SAMPLE, SPLIT_SUBJECT, describe_split, subject_grouped_split
from validation_utils import evaluate_baselines, evaluate_moe

RESULTS_DIR = Path("results")
SAMPLE_MOE_FILE = RESULTS_DIR / "moe_en_results.csv"
SAMPLE_BASELINE_FILE = RESULTS_DIR / "baseline_results.csv"
SUBJECT_MOE_FILE = RESULTS_DIR / "moe_en_results_subject.csv"
SUBJECT_BASELINE_FILE = RESULTS_DIR / "baseline_results_subject.csv"
SUBJECT_PRED_FILE = RESULTS_DIR / "test_predictions_subject.csv"
SPLIT_INFO_FILE = RESULTS_DIR / "split_info.csv"
COMPARISON_FILE = RESULTS_DIR / "validation_comparison.csv"


def load_sample_results():
    """Load existing sample-stratified metrics with split label."""
    rows = []
    if SAMPLE_MOE_FILE.exists():
        moe = pd.read_csv(SAMPLE_MOE_FILE)
        moe["split"] = SPLIT_SAMPLE
        rows.append(moe)
    if SAMPLE_BASELINE_FILE.exists():
        base = pd.read_csv(SAMPLE_BASELINE_FILE)
        base["split"] = SPLIT_SAMPLE
        rows.append(base)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def add_sample_split_info(df, meta):
    """Add train/test counts to sample-level results for comparison table."""
    from split_utils import sample_stratified_split

    train_idx, test_idx = sample_stratified_split(meta)
    info = describe_split(meta, train_idx, test_idx, SPLIT_SAMPLE)
    for key in ("n_train_samples", "n_test_samples", "n_train_subjects", "n_test_subjects", "n_overlap_subjects"):
        if key not in df.columns:
            df[key] = info[key]
    return df


def main():
    parser = argparse.ArgumentParser(description="Subject-grouped validation")
    parser.add_argument("--baselines-only", action="store_true", help="Skip MoE (faster)")
    parser.add_argument("--moe-only", action="store_true", help="Skip baselines")
    args = parser.parse_args()

    run_baselines = not args.moe_only
    run_moe = not args.baselines_only

    print("=" * 60, flush=True)
    print("Subject-Grouped Internal Validation", flush=True)
    print("=" * 60, flush=True)

    expr, meta = load_processed_data()
    X = expr.values.astype(float)
    y = meta["AGE_MID"].values.astype(float)

    train_idx, test_idx = subject_grouped_split(meta)
    split_info = describe_split(meta, train_idx, test_idx, SPLIT_SUBJECT)

    print(
        f"Subject split: {split_info['n_train_samples']:,} train samples "
        f"({split_info['n_train_subjects']:,} subjects), "
        f"{split_info['n_test_samples']:,} test samples "
        f"({split_info['n_test_subjects']:,} subjects)",
        flush=True,
    )
    print(f"Donor overlap: {split_info['n_overlap_subjects']} (must be 0)", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame([split_info]).to_csv(SPLIT_INFO_FILE, index=False)

    subject_rows = []
    pred_frames = []

    if run_baselines:
        print("\n--- Baselines (subject-grouped) ---", flush=True)
        base_rows, base_preds = evaluate_baselines(
            X, y, meta, train_idx, test_idx, SPLIT_SUBJECT
        )
        subject_rows.extend(base_rows)
        pred_frames.append(base_preds)
        pd.DataFrame(base_rows).to_csv(SUBJECT_BASELINE_FILE, index=False)
        print(f"Saved -> {SUBJECT_BASELINE_FILE}", flush=True)

    if run_moe:
        print("\n--- MoE (subject-grouped) ---", flush=True)
        moe_rows, moe_preds, _, _ = evaluate_moe(
            X, y, meta, train_idx, test_idx, SPLIT_SUBJECT
        )
        subject_rows.extend(moe_rows)
        pred_frames.append(moe_preds)
        pd.DataFrame(moe_rows).to_csv(SUBJECT_MOE_FILE, index=False)
        print(f"Saved -> {SUBJECT_MOE_FILE}", flush=True)

    if pred_frames:
        preds = pd.concat(pred_frames, ignore_index=True)
        preds.to_csv(SUBJECT_PRED_FILE, index=False)
        print(f"Saved -> {SUBJECT_PRED_FILE}", flush=True)

    sample_df = load_sample_results()
    if not sample_df.empty:
        sample_df = add_sample_split_info(sample_df, meta)

    subject_df = pd.DataFrame(subject_rows)
    comparison = pd.concat([sample_df, subject_df], ignore_index=True)
    comparison.to_csv(COMPARISON_FILE, index=False)

    print("\n" + "=" * 60, flush=True)
    print("Validation Comparison", flush=True)
    print("=" * 60, flush=True)
    cols = ["model", "split", "mae", "r2", "pearson_r", "n_test_samples", "n_overlap_subjects"]
    cols = [c for c in cols if c in comparison.columns]
    print(comparison[cols].to_string(index=False), flush=True)
    print(f"\nSaved -> {COMPARISON_FILE}", flush=True)


if __name__ == "__main__":
    main()
