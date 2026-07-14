"""Shared training and evaluation for validation splits."""

import gc
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from moe_utils import add_routes, evaluate, predict_moe, train_experts
from split_utils import describe_split

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}


def _metrics_row(model_name, split_name, split_info, y_true, y_pred, train_time_sec=None):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    row = {
        "model": model_name,
        "split": split_name,
        "mae": mae,
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p,
        **split_info,
    }
    if train_time_sec is not None:
        row["train_time_sec"] = train_time_sec
    return row


def evaluate_baselines(X, y, meta, train_idx, test_idx, split_name):
    """Train baseline models on one split and return metrics + predictions."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    meta_test = meta.iloc[test_idx].reset_index()
    split_info = describe_split(meta, train_idx, test_idx, split_name)

    models = {
        "Elastic Net": ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], cv=5),
        "Random Forest": RandomForestRegressor(**RF_PARAMS),
        "XGBoost": XGBRegressor(**XGB_PARAMS),
    }

    results = []
    pred_frames = []

    for name, model in models.items():
        print(f"  Training baseline {name} ({split_name})...", flush=True)
        if name == "Elastic Net":
            scaler = StandardScaler()
            curr_X_train = scaler.fit_transform(X_train)
            curr_X_test = scaler.transform(X_test)
        else:
            curr_X_train, curr_X_test = X_train, X_test

        start = time.time()
        model.fit(curr_X_train, y_train)
        train_time = time.time() - start
        y_pred = model.predict(curr_X_test)

        results.append(_metrics_row(name, split_name, split_info, y_test, y_pred, train_time))
        pred = meta_test[["SAMPID", "SUBJID", "AGE_MID", "SMTSD"]].copy()
        pred["PRED_AGE"] = y_pred
        pred["AGE_GAP"] = pred["PRED_AGE"] - pred["AGE_MID"]
        pred["model"] = name
        pred["split"] = split_name
        pred_frames.append(pred)

        del model
        gc.collect()

    return results, pd.concat(pred_frames, ignore_index=True)


def evaluate_moe(X, y, meta, train_idx, test_idx, split_name):
    """Train MoE on one split and return metrics + predictions."""
    meta = add_routes(meta)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    meta_train = meta.iloc[train_idx]
    meta_test = meta.iloc[test_idx]
    split_info = describe_split(meta, train_idx, test_idx, split_name)

    print(f"  Training MoE experts ({split_name})...", flush=True)
    start = time.time()
    experts, scalers = train_experts(X_train, y_train, meta_train["ROUTE"].values)
    train_time = time.time() - start

    preds = predict_moe(X_test, meta_test["ROUTE"].values, experts, scalers)
    row = _metrics_row("MoE_EN", split_name, split_info, y_test, preds, train_time)

    pred = meta_test.reset_index()[["SAMPID", "SUBJID", "AGE_MID", "SMTSD", "ROUTE"]].copy()
    pred["PRED_AGE"] = preds
    pred["AGE_GAP"] = pred["PRED_AGE"] - pred["AGE_MID"]
    pred["split"] = split_name

    return [row], pred, experts, scalers
