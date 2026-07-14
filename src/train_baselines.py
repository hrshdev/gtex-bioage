"""
Baseline Age Estimation Models
================================================================
Trained on GTEx v11 processed (Top 5000 variance genes).
Implements Elastic Net, Random Forest, and XGBoost.
"""

import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from data_utils import load_processed_data, make_age_bins

# Configuration
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "baseline_results.csv"
# Elastic Net: Tuned via CV
# RF: Moderate depth to prevent overfitting on 5000 features
# XGBoost: Moderate learning rate and depth
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1
}
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1
}

def log(msg):
    print(f"[baselines] {msg}", flush=True)

def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)

def load_and_align_data():
    section("Stage 1: Loading and Aligning Data")
    expr, meta = load_processed_data()
    X = expr.values.astype(np.float32)
    y = meta["AGE_MID"].values.astype(np.float32)
    log(f"Aligned samples: {len(meta):,}")
    log(f"Features (genes): {expr.shape[1]:,}")
    return X, y, meta


def train_and_evaluate(X, y, meta):
    section("Stage 2: Training and Evaluation")

    meta = meta.copy()
    meta["AGE_BIN"] = make_age_bins(meta)

    # Stratify by Age Bin to ensure balanced age distribution in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=meta['AGE_BIN']
    )

    log(f"Train set: {len(X_train):,}")
    log(f"Test set: {len(X_test):,}")

    results = []

    # Define models to iterate through
    models = {
        "Elastic Net": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5),
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "XGBoost": XGBRegressor(**XGB_PARAMS)
    }

    for name, model in models.items():
        log(f"Training {name}...")

        # Scale data for linear models (Elastic Net)
        if name == "Elastic Net":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            curr_X_train, curr_X_test = X_train_scaled, X_test_scaled
        else:
            curr_X_train, curr_X_test = X_train, X_test

        start_time = time.time()
        model.fit(curr_X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(curr_X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r, p = pearsonr(y_test, y_pred)

        log(f"  {name} -> MAE: {mae:.2f}, R2: {r2:.2f}, Pearson: {r:.2f}")

        results.append({
            "model": name,
            "mae": mae,
            "r2": r2,
            "pearson_r": r,
            "pearson_p": p,
            "train_time_sec": train_time
        })

        # Free memory
        del model
        gc.collect()

    return results

def main():
    log("=" * 60)
    log("Baseline Age Estimation Training")
    log("=" * 60)

    try:
        X, y, meta = load_and_align_data()
        results = train_and_evaluate(X, y, meta)

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_FILE, index=False)

        section("Final Baseline Results")
        print(df_results.to_string(index=False))

        log(f"Results saved to -> {RESULTS_FILE}")

    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    log("=" * 60)

if __name__ == "__main__":
    main()
