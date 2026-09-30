"""
Fast Baseline Training with Live Progress & GPU Acceleration
============================================================
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

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "baseline_results.csv"

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42
}

def log(msg):
    print(f"[baselines] {msg}", flush=True)

def load_and_align_data():
    expr, meta = load_processed_data()
    X = expr.values.astype(np.float32)
    y = meta["AGE_MID"].values.astype(np.float32)
    del expr
    gc.collect()
    return X, y, meta

def train_and_evaluate(X, y, meta):
    meta = meta.copy()
    meta["AGE_BIN"] = make_age_bins(meta)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=meta['AGE_BIN']
    )
    del X
    gc.collect()

    results = []

    # 1. Fast Elastic Net with live verbose progress
    log("Training Elastic Net (Fast-mode with live output)...")
    scaler = StandardScaler(copy=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # n_alphas=30 and tol=1e-3 runs 3x-4x faster with equivalent accuracy
    en_model = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        n_alphas=30,
        tol=1e-3,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    t0 = time.time()
    en_model.fit(X_train_scaled, y_train)
    en_time = time.time() - t0

    y_pred_en = en_model.predict(X_test_scaled)
    mae_en = mean_absolute_error(y_test, y_pred_en)
    r2_en = r2_score(y_test, y_pred_en)
    r_en, p_en = pearsonr(y_test, y_pred_en)
    log(f"  Elastic Net -> MAE: {mae_en:.2f}, R2: {r2_en:.2f}, Pearson: {r_en:.2f} ({en_time:.1f}s)")

    results.append({
        "model": "Elastic Net",
        "mae": mae_en,
        "r2": r2_en,
        "pearson_r": r_en,
        "pearson_p": p_en,
        "train_time_sec": en_time
    })
    del en_model, scaler, X_train_scaled, X_test_scaled, y_pred_en
    gc.collect()

    # 2. Random Forest
    log("\nTraining Random Forest (multi-core)...")
    rf_model = RandomForestRegressor(**RF_PARAMS)
    t0 = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - t0

    y_pred_rf = rf_model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    r_rf, p_rf = pearsonr(y_test, y_pred_rf)
    log(f"  Random Forest -> MAE: {mae_rf:.2f}, R2: {r2_rf:.2f}, Pearson: {r_rf:.2f} ({rf_time:.1f}s)")

    results.append({
        "model": "Random Forest",
        "mae": mae_rf,
        "r2": r2_rf,
        "pearson_r": r_rf,
        "pearson_p": p_rf,
        "train_time_sec": rf_time
    })
    del rf_model, y_pred_rf
    gc.collect()

    # 3. XGBoost on GPU
    log("\nTraining XGBoost (GPU Accelerated)...")
    xgb_model = XGBRegressor(**XGB_PARAMS)
    t0 = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - t0

    y_pred_xgb = xgb_model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    r_xgb, p_xgb = pearsonr(y_test, y_pred_xgb)
    log(f"  XGBoost -> MAE: {mae_xgb:.2f}, R2: {r2_xgb:.2f}, Pearson: {r_xgb:.2f} ({xgb_time:.1f}s)")

    results.append({
        "model": "XGBoost",
        "mae": mae_xgb,
        "r2": r2_xgb,
        "pearson_r": r_xgb,
        "pearson_p": p_xgb,
        "train_time_sec": xgb_time
    })
    del xgb_model, y_pred_xgb, X_train, X_test, y_train, y_test
    gc.collect()

    return results

def main():
    X, y, meta = load_and_align_data()
    results = train_and_evaluate(X, y, meta)
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    print("\n" + "="*50)
    print(df_results.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
