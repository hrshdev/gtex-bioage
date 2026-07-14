"""
Standalone MoE inference on the processed GTEx test set.
Loads saved experts from models/ and writes test_predictions.csv.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from data_utils import load_processed_data
from moe_utils import add_routes, evaluate, predict_moe, stratified_split

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"


def load_models():
    experts = {
        "neural": joblib.load(MODELS_DIR / "neural_expert.joblib"),
        "non_neural": joblib.load(MODELS_DIR / "non_neural_expert.joblib"),
    }
    scalers = {
        "neural": joblib.load(MODELS_DIR / "neural_scaler.joblib"),
        "non_neural": joblib.load(MODELS_DIR / "non_neural_scaler.joblib"),
    }
    return experts, scalers


def main():
    parser = argparse.ArgumentParser(description="Run MoE inference on processed GTEx data")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score all samples instead of the stratified test split only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PREDICTIONS_FILE,
        help="Output CSV path for predictions",
    )
    args = parser.parse_args()

    for path in [
        MODELS_DIR / "neural_expert.joblib",
        MODELS_DIR / "non_neural_expert.joblib",
        MODELS_DIR / "neural_scaler.joblib",
        MODELS_DIR / "non_neural_scaler.joblib",
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}\nRun: python src/train_moe_en.py")

    expr, meta = load_processed_data()
    meta = add_routes(meta)
    experts, scalers = load_models()

    if args.all:
        X = expr.values.astype(np.float32)
        meta_pred = meta
        y_true = meta["AGE_MID"].values.astype(np.float32)
        split_label = "all samples"
    else:
        _, test_idx = stratified_split(meta)
        X = expr.iloc[test_idx].values.astype(np.float32)
        meta_pred = meta.iloc[test_idx]
        y_true = meta_pred["AGE_MID"].values.astype(np.float32)
        split_label = "test split"

    preds = predict_moe(X, meta_pred["ROUTE"].values, experts, scalers)
    metrics = evaluate(y_true, preds)

    print(f"Predicting on {len(meta_pred):,} samples ({split_label})")
    print(
        f"MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.2f}, "
        f"Pearson: {metrics['pearson_r']:.2f}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = meta_pred.reset_index()
    out["PRED_AGE"] = preds
    out["AGE_GAP"] = out["PRED_AGE"] - out["AGE_MID"]
    out[
        ["SAMPID", "SUBJID", "AGE_MID", "PRED_AGE", "AGE_GAP", "SMTSD", "ROUTE"]
    ].to_csv(args.output, index=False)

    print(f"Predictions saved -> {args.output}")


if __name__ == "__main__":
    main()
