"""Shared MoE router, training, prediction, and evaluation helpers."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from split_utils import sample_stratified_split

NEURAL_TISSUES = ["brain", "spinal cord", "nerve"]
ROUTES = ["neural", "non_neural"]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]


def get_route(tissue):
    """Assign sample to neural or non_neural expert based on tissue name."""
    if pd.isna(tissue):
        return "non_neural"
    tissue_lower = str(tissue).lower()
    return "neural" if any(kw in tissue_lower for kw in NEURAL_TISSUES) else "non_neural"


def add_routes(meta):
    """Add ROUTE column from SMTSD."""
    meta = meta.copy()
    meta["ROUTE"] = meta["SMTSD"].apply(get_route)
    return meta


def stratified_split(meta, test_size=0.2, random_state=42):
    """Return train/test indices with age-stratified sample-level split."""
    return sample_stratified_split(meta, test_size=test_size, random_state=random_state)


def _fit_expert_cv(X_scaled, y_exp, route, show_progress=True):
    """Fit ElasticNetCV with per-l1_ratio progress bar."""
    best_model = None
    best_score = np.inf
    iterator = tqdm(
        L1_RATIOS,
        desc=f"  {route} expert",
        unit="l1_ratio",
        disable=not show_progress,
    )
    for l1_ratio in iterator:
        model = ElasticNetCV(
            l1_ratio=[l1_ratio],
            cv=5,
            random_state=42,
            n_jobs=-1,
            max_iter=5000,
        )
        model.fit(X_scaled, y_exp)
        score = float(model.mse_path_.min())
        if score < best_score:
            best_score = score
            best_model = model
        if show_progress:
            iterator.set_postfix(l1=l1_ratio, alpha=f"{model.alpha_:.3g}", mse=f"{score:.2f}")
    return best_model


def train_experts(X_train, y_train, routes_train, show_progress=True):
    """Train Elastic Net expert per route. Returns (experts, scalers) dicts."""
    experts = {}
    scalers = {}

    route_iter = tqdm(ROUTES, desc="MoE experts", unit="expert", disable=not show_progress)
    for route in route_iter:
        mask = routes_train == route
        X_exp = X_train[mask]
        y_exp = y_train[mask]
        if show_progress:
            route_iter.set_postfix(route=route, samples=len(X_exp))
        else:
            print(f"  Training {route} expert on {len(X_exp):,} samples...", flush=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_exp)
        experts[route] = _fit_expert_cv(X_scaled, y_exp, route, show_progress=show_progress)
        scalers[route] = scaler

    return experts, scalers


def predict_moe(X, routes, experts, scalers):
    """Route samples to experts and return age predictions."""
    preds = np.zeros(len(X), dtype=np.float32)
    for route in ROUTES:
        mask = routes == route
        if not mask.any() or route not in experts:
            continue
        X_scaled = scalers[route].transform(X[mask])
        preds[mask] = experts[route].predict(X_scaled)
    return preds


def evaluate(y_true, y_pred):
    """Return MAE, R2, Pearson r and p."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    return {"mae": mae, "r2": r2, "pearson_r": r, "pearson_p": p}


def export_coefficients(experts, gene_names, output_path, top_n=100):
    """Save per-gene coefficients and top-weighted genes per expert."""
    rows = []
    for route, model in experts.items():
        for gene, coef in zip(gene_names, model.coef_):
            rows.append(
                {"expert": route, "ENSG": gene, "coefficient": coef, "intercept": model.intercept_}
            )

    df = pd.DataFrame(rows)
    df["abs_coefficient"] = df["coefficient"].abs()
    df = df.sort_values(["expert", "abs_coefficient"], ascending=[True, False])
    df.to_csv(output_path, index=False)

    print(f"Saved coefficients -> {output_path}")
    for route in ROUTES:
        top = df[df["expert"] == route].head(top_n)
        print(f"  Top gene ({route}): {top.iloc[0]['ENSG']} (coef={top.iloc[0]['coefficient']:.4f})")
