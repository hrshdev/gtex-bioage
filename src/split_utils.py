"""Train/test split helpers for sample-level and subject-grouped evaluation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from data_utils import make_age_bins

SPLIT_SAMPLE = "sample_stratified"
SPLIT_SUBJECT = "subject_grouped"


def sample_stratified_split(meta, test_size=0.2, random_state=42):
    """Age-stratified split at sample level (may leak donors across train/test)."""
    meta = meta.copy()
    meta["AGE_BIN"] = make_age_bins(meta)
    indices = np.arange(len(meta))
    return train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=meta["AGE_BIN"],
    )


def subject_grouped_split(meta, test_size=0.2, random_state=42):
    """Split by SUBJID so no donor appears in both train and test."""
    meta = meta.copy()
    meta["AGE_BIN"] = make_age_bins(meta)
    subj = meta.groupby("SUBJID", as_index=False).agg(AGE_BIN=("AGE_BIN", "first"))

    try:
        train_subj, test_subj = train_test_split(
            subj["SUBJID"],
            test_size=test_size,
            random_state=random_state,
            stratify=subj["AGE_BIN"],
        )
    except ValueError:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(
            gss.split(np.arange(len(meta)), groups=meta["SUBJID"].values)
        )
        verify_no_leakage(meta, train_idx, test_idx)
        return train_idx, test_idx

    train_mask = meta["SUBJID"].isin(train_subj).values
    test_mask = meta["SUBJID"].isin(test_subj).values
    indices = np.arange(len(meta))
    train_idx = indices[train_mask]
    test_idx = indices[test_mask]
    verify_no_leakage(meta, train_idx, test_idx)
    return train_idx, test_idx


def verify_no_leakage(meta, train_idx, test_idx):
    """Raise if any donor appears in both train and test."""
    train_subj = set(meta.iloc[train_idx]["SUBJID"])
    test_subj = set(meta.iloc[test_idx]["SUBJID"])
    overlap = train_subj & test_subj
    if overlap:
        raise ValueError(f"Donor leakage detected: {len(overlap)} subjects in both splits")


def describe_split(meta, train_idx, test_idx, split_name):
    """Return summary stats for logging and CSV export."""
    train_meta = meta.iloc[train_idx]
    test_meta = meta.iloc[test_idx]
    return {
        "split": split_name,
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "n_train_subjects": train_meta["SUBJID"].nunique(),
        "n_test_subjects": test_meta["SUBJID"].nunique(),
        "n_overlap_subjects": len(set(train_meta["SUBJID"]) & set(test_meta["SUBJID"])),
    }
