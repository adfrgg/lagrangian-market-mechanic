"""Experiment runner for Order-Book Lagrangian Market Mechanics v1.0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lagrangian_market.l2_features import read_table
from lagrangian_market.l2_models import (
    compute_l2_incremental_value,
    l2_coefficient_table,
    run_l2_fold_metrics,
    run_l2_model_suite,
    train_test_split_time,
)
from lagrangian_market.l2_plots import (
    plot_coefficient_stability,
    plot_fold_delta_r2,
    plot_observed_vs_predicted_force,
    plot_orderbook_potential_snapshot,
    plot_potential_force_timeseries,
    plot_price_mid_spread,
    plot_residual_added_value,
)


L2_REQUIRED_COLUMNS = [
    "F_next",
    "p",
    "u",
    "F_pot_1bps",
    "F_pot_2bps",
    "F_pot_5bps",
    "F_pot_10bps",
    "F_flow",
    "book_imbalance_20",
    "spread_bps",
]


def load_l2_dataset(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
    missing = [col for col in L2_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required L2 columns: {missing}")
    return df.reset_index(drop=True)


def run_l2_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    folds: int = 5,
) -> pd.DataFrame:
    """Run L2 model suite, folds, reports, and plots."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_l2_dataset(data_path)
    clean = df.dropna(subset=L2_REQUIRED_COLUMNS).copy()
    if len(clean) < 20:
        raise ValueError(f"Insufficient clean L2 samples: {len(clean)}")

    metrics = run_l2_model_suite(clean, train_ratio=train_ratio)
    incremental = compute_l2_incremental_value(metrics)
    coefficients = l2_coefficient_table(metrics)
    fold_metrics, fold_summary = run_l2_fold_metrics(clean, folds=folds)

    metrics.to_csv(output_dir / "metrics_all.csv", index=False)
    incremental.to_csv(output_dir / "incremental_value.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False)

    train_df, test_df = train_test_split_time(clean, train_ratio=train_ratio)
    features = ["F_pot_5bps", "F_flow", "book_imbalance_20", "p", "u", "spread_bps"]
    model = LinearRegression().fit(train_df[features], train_df["F_next"])
    pred = model.predict(test_df[features])

    plot_price_mid_spread(clean, figures_dir / "price_mid_spread.png")
    plot_potential_force_timeseries(clean, figures_dir / "potential_force_timeseries.png")
    plot_observed_vs_predicted_force(test_df, pred, figures_dir / "observed_vs_predicted_force.png")
    plot_residual_added_value(clean, figures_dir / "residual_added_value.png")
    plot_fold_delta_r2(fold_metrics, figures_dir / "fold_delta_R2.png")
    plot_coefficient_stability(fold_metrics, figures_dir / "coefficient_stability.png")
    plot_orderbook_potential_snapshot(clean, figures_dir / "orderbook_potential_snapshot.png")

    _write_report(output_dir / "report.md", data_path, clean, metrics, incremental, fold_summary)
    print("L2 experiment complete")
    print(f"Rows: {len(clean)}")
    print(f"Output: {output_dir}")
    print(metrics.to_string(index=False))
    return metrics


def _write_report(
    path: Path,
    data_path: str | Path,
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    best = metrics.sort_values("test_r2", ascending=False).iloc[0]
    full_inc = incremental.loc[incremental["comparison"] == "l2_full_vs_p_u"]
    full_text = full_inc.iloc[0].to_dict() if not full_inc.empty else {}
    robust = _robust_positive(incremental, fold_summary, len(df))
    report = f"""# Order-Book Lagrangian Market Mechanics v1.0 Report

Data: `{data_path}`

Rows after cleaning: {len(df)}

Best test R2 model: {best['model']} ({best['test_r2']:.6g})

## Primary Question

Does order-book potential `F_pot` add value beyond `p` and `u`?

`l2_full_vs_p_u`:

```text
{full_text}
```

## Incremental Value

```text
{incremental.to_string(index=False)}
```

## Fold Summary

```text
{fold_summary.to_string(index=False)}
```

## Robust Positive Check

{robust}

This is empirical mechanics validation only. It is not a trading strategy and
does not report trading returns.
"""
    path.write_text(report, encoding="utf-8")


def _robust_positive(incremental: pd.DataFrame, fold_summary: pd.DataFrame, sample_size: int) -> str:
    full = incremental.loc[incremental["comparison"] == "l2_full_vs_p_u"]
    if full.empty:
        return "l2_full comparison unavailable."
    row = full.iloc[0]
    fold_row = fold_summary.loc[fold_summary["model"] == "l2_full"]
    fold_rate = float(fold_row["positive_test_r2_rate"].iloc[0]) if not fold_row.empty else np.nan
    sign_stability = float(fold_row["fpot_coef_sign_stability"].iloc[0]) if not fold_row.empty and "fpot_coef_sign_stability" in fold_row else np.nan
    positive = (
        row["delta_test_R2"] > 0
        and row["delta_RMSE"] > 0
        and np.isfinite(fold_rate)
        and fold_rate >= 0.6
        and np.isfinite(sign_stability)
        and sign_stability >= 0.6
        and sample_size >= 1000
    )
    if positive:
        return "Robust-positive criteria passed."
    return (
        "Robust-positive criteria not fully met. Require l2_full > baseline_p_u "
        "on test R2 and RMSE, fold positive rate >= 0.6, F_pot coefficient sign "
        "stability >= 0.6, and sample size >= 1000."
    )
