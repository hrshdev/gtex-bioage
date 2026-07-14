"""
Mixture-of-Experts (MoE) Age Estimation
================================================================
Trained on GTEx v11 processed (Top 5000 variance genes).

Architecture: Rule-based router -> Specialized Elastic Net Experts.
- Neural Expert: Brain, Spinal Cord, Nerve
- Non-Neural Expert: All other tissues
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from data_utils import load_processed_data
from moe_utils import (
    add_routes,
    evaluate,
    export_coefficients,
    predict_moe,
    stratified_split,
    train_experts,
)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RESULTS_FILE = RESULTS_DIR / "moe_en_results.csv"
PREDICTIONS_FILE = RESULTS_DIR / "test_predictions.csv"
COEFFICIENTS_FILE = RESULTS_DIR / "moe_gene_coefficients.csv"


def main():
    print("--- Loading Data ---", flush=True)
    expr, meta = load_processed_data()
    X = expr.values.astype(np.float32)
    y = meta["AGE_MID"].values.astype(np.float32)
    gene_names = expr.columns.tolist()
    meta = add_routes(meta)

    train_idx, test_idx = stratified_split(meta)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    meta_train = meta.iloc[train_idx]
    meta_test = meta.iloc[test_idx]

    print(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples.")

    experts, scalers = train_experts(
        X_train, y_train, meta_train["ROUTE"].values
    )

    print("Evaluating MoE model...")
    final_preds = predict_moe(
        X_test, meta_test["ROUTE"].values, experts, scalers
    )

    metrics = evaluate(y_test, final_preds)
    print(
        f"Final Results -> MAE: {metrics['mae']:.2f}, "
        f"R2: {metrics['r2']:.2f}, Pearson: {metrics['pearson_r']:.2f}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    pd.DataFrame([{"model": "MoE_EN", **metrics}]).to_csv(RESULTS_FILE, index=False)

    for route in experts:
        joblib.dump(experts[route], MODELS_DIR / f"{route}_expert.joblib")
        joblib.dump(scalers[route], MODELS_DIR / f"{route}_scaler.joblib")

    export_coefficients(experts, gene_names, COEFFICIENTS_FILE)

    res_df = meta_test.reset_index()
    res_df["PRED_AGE"] = final_preds
    res_df["AGE_GAP"] = res_df["PRED_AGE"] - res_df["AGE_MID"]
    res_df[
        ["SAMPID", "SUBJID", "AGE_MID", "PRED_AGE", "AGE_GAP", "SMTSD", "ROUTE"]
    ].to_csv(PREDICTIONS_FILE, index=False)

    print(f"Results saved -> {RESULTS_FILE}")
    print(f"Predictions saved -> {PREDICTIONS_FILE}")
    print("Done! Models saved.")


if __name__ == "__main__":
    main()
