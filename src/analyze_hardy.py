"""Hardy scale validation and sensitivity analysis for MoE predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score

from data_utils import load_processed_data
from moe_utils import add_routes, evaluate, predict_moe, stratified_split
from plot_style import apply_thesis_style

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures" / "hardy"
THESIS_FIGURES_DIR = Path(r"C:\Research\Research\Thesis\figures")

HARDY_MAP = {
    0: "Ventilator",
    1: "Violent/Fast",
    2: "Fast illness",
    3: "Intermediate",
    4: "Slow illness",
}
HARDY_ORDER = ["Ventilator", "Violent/Fast", "Fast illness", "Intermediate", "Slow illness"]
HARDY_COLORS = {
    "Ventilator": "#4C6EF5",
    "Violent/Fast": "#F76707",
    "Fast illness": "#51CF66",
    "Intermediate": "#845EF7",
    "Slow illness": "#20C997",
}


def load_metadata_only() -> pd.DataFrame:
    meta_path = Path("data/processed/metadata_filtered.parquet")
    return pd.read_parquet(meta_path)


def load_predictions(path: Path) -> pd.DataFrame:
    pred = pd.read_csv(path)
    meta = load_metadata_only()
    meta_cols = [c for c in ["DTHHRDY", "SEX", "AGE"] if c in meta.columns]
    out = pred.merge(meta[meta_cols], left_on="SAMPID", right_index=True, how="left")
    if out["DTHHRDY"].isna().any():
        raise ValueError(f"Missing Hardy for {out['DTHHRDY'].isna().sum()} prediction rows")
    out["HARDY_LABEL"] = out["DTHHRDY"].map(HARDY_MAP)
    out["ABS_GAP"] = out["AGE_GAP"].abs()
    return out


def donor_level(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("SUBJID", as_index=False)
        .agg(
            DTHHRDY=("DTHHRDY", "first"),
            HARDY_LABEL=("HARDY_LABEL", "first"),
            AGE_MID=("AGE_MID", "first"),
            mean_gap=("AGE_GAP", "mean"),
            mean_abs_gap=("ABS_GAP", "mean"),
            n_samples=("SAMPID", "count"),
        )
    )


def mae_by_hardy(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for hardy in HARDY_ORDER:
        g = df[df["HARDY_LABEL"] == hardy]
        if g.empty:
            continue
        rows.append(
            {
                "split": label,
                "hardy": hardy,
                "n": len(g),
                "mae": mean_absolute_error(g["AGE_MID"], g["PRED_AGE"]),
                "r2": r2_score(g["AGE_MID"], g["PRED_AGE"]),
                "mean_gap": g["AGE_GAP"].mean(),
                "sd_gap": g["AGE_GAP"].std(),
                "mean_abs_gap": g["ABS_GAP"].mean(),
            }
        )
    return pd.DataFrame(rows)


def metrics_row(scenario: str, df: pd.DataFrame) -> dict:
    return {
        "scenario": scenario,
        "n_samples": len(df),
        "n_donors": df["SUBJID"].nunique(),
        "mae": mean_absolute_error(df["AGE_MID"], df["PRED_AGE"]),
        "r2": r2_score(df["AGE_MID"], df["PRED_AGE"]),
        "mean_gap": df["AGE_GAP"].mean(),
        "mean_abs_gap": df["ABS_GAP"].mean(),
    }


def fast_retrain_without_ventilator(models_dir: Path) -> dict:
    """Retrain MoE on Hardy != 0 using fixed hyperparameters from the primary model."""
    expr, meta = load_processed_data()
    meta = add_routes(meta)
    mask = meta["DTHHRDY"] != 0
    expr = expr.loc[mask]
    meta = meta.loc[mask]

    X = expr.values.astype(np.float32)
    y = meta["AGE_MID"].values.astype(np.float32)
    train_idx, test_idx = stratified_split(meta)

    experts = {}
    scalers = {}
    for route in ("neural", "non_neural"):
        saved = joblib.load(models_dir / f"{route}_expert.joblib")
        train_mask = meta.iloc[train_idx]["ROUTE"].values == route
        X_exp = X[train_idx][train_mask]
        y_exp = y[train_idx][train_mask]
        scaler = StandardScaler().fit(X_exp)
        model = ElasticNet(
            alpha=saved.alpha_,
            l1_ratio=float(saved.l1_ratio_),
            max_iter=10000,
            random_state=42,
        )
        model.fit(scaler.transform(X_exp), y_exp)
        experts[route] = model
        scalers[route] = scaler

    routes_test = meta.iloc[test_idx]["ROUTE"].values
    preds = predict_moe(X[test_idx], routes_test, experts, scalers)
    metrics = evaluate(y[test_idx], preds)
    return {
        "scenario": "retrained_excluding_ventilator",
        "n_samples": len(test_idx),
        "n_donors": meta.iloc[test_idx]["SUBJID"].nunique(),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        **metrics,
        "mean_gap": float(np.mean(preds - y[test_idx])),
        "mean_abs_gap": float(np.mean(np.abs(preds - y[test_idx]))),
    }


def plot_gap_by_hardy(df: pd.DataFrame, out_path: Path):
    apply_thesis_style()
    data = [df.loc[df["HARDY_LABEL"] == h, "AGE_GAP"].values for h in HARDY_ORDER]
    labels = [h for h in HARDY_ORDER if (df["HARDY_LABEL"] == h).any()]
    data = [df.loc[df["HARDY_LABEL"] == h, "AGE_GAP"].values for h in labels]
    colors = [HARDY_COLORS[h] for h in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
        body.set_edgecolor("white")

    bp = ax.boxplot(
        data,
        labels=labels,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Age gap (predicted − chronological, years)")
    ax.set_title("MoE age gap by Hardy scale (sample-stratified test set)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_mae_by_hardy(summary: pd.DataFrame, out_path: Path, overall_mae: float):
    apply_thesis_style()
    summary = summary.sort_values(
        "hardy", key=lambda s: s.map({h: i for i, h in enumerate(HARDY_ORDER)})
    )
    colors = [HARDY_COLORS[h] for h in summary["hardy"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(summary["hardy"], summary["mae"], color=colors, edgecolor="white")
    ax.axhline(
        overall_mae,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall test MAE ({overall_mae:.2f} yr)",
    )
    ax.set_ylabel("MAE (years)")
    ax.set_title("MoE error by Hardy scale category", fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=20)
    ymax = max(summary["mae"].max(), overall_mae)
    ax.set_ylim(0, ymax * 1.22)
    for bar, n, mae in zip(bars, summary["n"], summary["mae"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{mae:.2f}\n(n={n:,})",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def copy_to_thesis(src: Path, name: str):
    dest = THESIS_FIGURES_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    print(f"Copied -> {dest}")


def main(retrain: bool = False):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sample_df = load_predictions(RESULTS_DIR / "test_predictions.csv")

    hardy_summary = mae_by_hardy(sample_df, "sample_stratified")
    hardy_summary.to_csv(RESULTS_DIR / "hardy_prediction_summary.csv", index=False)

    sensitivity_rows = [
        metrics_row("primary_all_test", sample_df),
        metrics_row("primary_non_ventilator_test", sample_df[sample_df["DTHHRDY"] != 0]),
        metrics_row("primary_ventilator_only_test", sample_df[sample_df["DTHHRDY"] == 0]),
    ]
    if retrain:
        print("Fast retrain on non-ventilator cohort (fixed hyperparameters)...")
        sensitivity_rows.append(fast_retrain_without_ventilator(Path("models")))
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(RESULTS_DIR / "hardy_sensitivity_summary.csv", index=False)

    plot_gap_by_hardy(sample_df, FIGURES_DIR / "01_gap_by_hardy.png")
    overall_mae = metrics_row("primary_all_test", sample_df)["mae"]
    plot_mae_by_hardy(hardy_summary, FIGURES_DIR / "02_mae_by_hardy.png", overall_mae)
    copy_to_thesis(FIGURES_DIR / "01_gap_by_hardy.png", "figure4.13.png")
    copy_to_thesis(FIGURES_DIR / "02_mae_by_hardy.png", "figure4.12.png")

    donor = donor_level(sample_df)
    kw_stat, kw_p = stats.kruskal(
        *[g["ABS_GAP"].values for _, g in sample_df.groupby("DTHHRDY") if len(g) >= 5]
    )
    donor_kw_stat, donor_kw_p = stats.kruskal(
        *[g["mean_gap"].values for _, g in donor.groupby("DTHHRDY") if len(g) >= 3]
    )

    print("\nHardy summary by category:")
    print(hardy_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSensitivity scenarios:")
    print(sensitivity.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nKruskal-Wallis |age gap| by Hardy (samples): p={kw_p:.4g}")
    print(f"Kruskal-Wallis mean gap by Hardy (donors): p={donor_kw_p:.4g}")
    print(f"\nSaved -> {RESULTS_DIR / 'hardy_prediction_summary.csv'}")
    print(f"Saved -> {RESULTS_DIR / 'hardy_sensitivity_summary.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Also retrain MoE on non-ventilator samples (slow; optional)",
    )
    args = parser.parse_args()
    main(retrain=args.retrain)
