"""End-to-end experiment runner for Lagrangian Market Mechanics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lagrangian_market.data import load_ohlcv
from lagrangian_market.evaluation import (
    coefficient_table_from_results,
    compute_incremental_value,
    run_model_suite,
)
from lagrangian_market.liquidity import add_liquidity_force
from lagrangian_market.mechanics import add_mechanics_features
from lagrangian_market.models import fit_linear_model, predict_model, train_test_split_time
from lagrangian_market.plots import (
    plot_actual_vs_predicted,
    plot_fliq_residual_added_value,
    plot_force_scatter,
    plot_inside_outside_force_scatter,
    plot_kinetic_activity,
    plot_momentum,
    plot_profile_diagnostics,
    plot_residuals,
    plot_target_vs_negative_p,
    plot_volume_profile_snapshot,
)


def _format_date_range(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns or df.empty:
        return "unknown"
    return f"{df['timestamp'].min()} to {df['timestamp'].max()}"


def _safe_row(metrics: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = metrics.loc[metrics["model"] == model_name]
    return rows.iloc[0] if not rows.empty else None


def _run_suite_or_empty(df: pd.DataFrame, train_ratio: float, min_samples: int) -> pd.DataFrame:
    if len(df) < min_samples:
        return pd.DataFrame({"status": [f"insufficient samples: {len(df)} < {min_samples}"]})
    return run_model_suite(df, train_ratio=train_ratio)


def _write_report(
    report_path: Path,
    feature_df: pd.DataFrame,
    metrics_all: pd.DataFrame,
    metrics_inside: pd.DataFrame,
    metrics_outside: pd.DataFrame,
    incremental: pd.DataFrame,
    volume_window: int,
    profile_window: int,
    bins: int,
    potential_mode: str,
    profile_method: str,
) -> None:
    best_r2 = metrics_all.sort_values("test_r2", ascending=False).iloc[0]
    best_rmse = metrics_all.sort_values("rmse", ascending=True).iloc[0]
    neg_p = _safe_row(metrics_all, "baseline_negative_p_fixed")
    liq_p = incremental.loc[incremental["comparison"] == "liq_p_vs_p"]
    liq_p_u = incremental.loc[incremental["comparison"] == "liq_p_u_vs_p_u"]
    main_increment = liq_p_u.iloc[0] if not liq_p_u.empty else (liq_p.iloc[0] if not liq_p.empty else None)
    outside_pct = float(feature_df["outside_profile_range"].mean() * 100.0) if "outside_profile_range" in feature_df else np.nan

    if main_increment is not None:
        inc_text = (
            f"After p/u controls, liquidity conclusion: {main_increment['conclusion']}. "
            f"Delta test R2 = {main_increment['delta_test_R2']:.6g}; "
            f"delta RMSE = {main_increment['delta_RMSE']:.6g}."
        )
    else:
        inc_text = "Incremental liquidity comparison was not available."

    if potential_mode == "barrier":
        mode_text = "Barrier mode treats high historical volume as higher potential or resistance."
    else:
        mode_text = "Well mode treats high historical volume as lower potential or an attractor/fair-value zone."

    neg_p_text = (
        f"Test R2 {neg_p['test_r2']:.6g}, RMSE {neg_p['rmse']:.6g}, sign accuracy {neg_p['sign_accuracy_raw']:.4f}."
        if neg_p is not None
        else "Not available."
    )

    outside_summary = (
        metrics_outside.to_string(index=False)
        if "model" in metrics_outside.columns
        else metrics_outside.to_string(index=False)
    )

    report = f"""# Lagrangian Market Mechanics v0.2 Report

## Dataset

- Rows after feature construction: {len(feature_df)}
- Date range: {_format_date_range(feature_df)}
- Close price range: {feature_df['close'].min():.6g} to {feature_df['close'].max():.6g}
- Volume normalization window: {volume_window}
- Liquidity profile window: {profile_window}
- Histogram bins: {bins}
- Potential mode: {potential_mode}
- Profile method: {profile_method}
- Outside profile range: {outside_pct:.2f}%

## Research Audit Framing

The target is `F_next_t = p_(t+1) - p_t`. Because this target mechanically
contains `-p_t`, a model using `u_t` or variables correlated with `p_t` can look
like it discovered damping even when it mostly learned target arithmetic. v0.2
therefore tests liquidity force after explicit momentum controls.

## Best Models

- Best by test R2: {best_r2['model']} with test R2 {best_r2['test_r2']:.6g}
- Best by RMSE: {best_rmse['model']} with RMSE {best_rmse['rmse']:.6g}
- Mechanical `-p_t` baseline: {neg_p_text}

## Incremental Liquidity Value

{inc_text}

Full incremental table:

```text
{incremental.to_string(index=False)}
```

## Inside vs Outside Profile Range Diagnostics

Inside-profile metrics:

```text
{metrics_inside.to_string(index=False)}
```

Outside-profile metrics:

```text
{outside_summary}
```

If performance is materially better inside the profile range, the potential is
more useful in liquidity-zone or mean-reverting regimes. If outside performance
is better, the force may be capturing breakout behavior. If both fail, the OHLCV
potential proxy is likely weak.

## Barrier / Well Interpretation

{mode_text}

Barrier and well modes can have identical linear-regression performance because
one is the sign inverse of the other. The coefficient sign must be interpreted
with the selected mode.

## Limitations

- This is not a trading strategy and does not produce buy/sell signals.
- OHLCV volume profile is not true order book liquidity.
- Historical executed volume is not the same as current resting depth.
- Histogram gradients are noisy and sensitive to bins/window.
- Chronological out-of-sample improvement beyond `-p_t` and p-only baselines is
  required before claiming liquidity potential adds information.
"""
    report_path.write_text(report, encoding="utf-8")


def _run_single_experiment(
    csv_path: str,
    output_dir: str | Path,
    volume_window: int,
    profile_window: int,
    bins: int,
    potential_mode: str,
    profile_method: str,
    train_ratio: float,
    min_samples: int,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    reports_path = output_path / "reports"
    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(csv_path)
    df = add_mechanics_features(df, volume_window=volume_window, use_log_price=True)
    df = add_liquidity_force(
        df,
        profile_window=profile_window,
        bins=bins,
        normalize=True,
        mode=potential_mode,
        profile_method=profile_method,
    )
    required = ["x", "u", "m", "p", "K", "F_obs", "F_next", "F_liq", "outside_profile_range"]
    feature_df = df.dropna(subset=required).copy()
    if len(feature_df) < min_samples:
        raise ValueError(f"Not enough valid rows after feature construction: {len(feature_df)} < {min_samples}")

    metrics_all = run_model_suite(feature_df, train_ratio=train_ratio)
    inside_df = feature_df.loc[~feature_df["outside_profile_range"].astype(bool)].copy()
    outside_df = feature_df.loc[feature_df["outside_profile_range"].astype(bool)].copy()
    metrics_inside = _run_suite_or_empty(inside_df, train_ratio, min_samples)
    metrics_outside = _run_suite_or_empty(outside_df, train_ratio, min_samples)
    incremental = compute_incremental_value(metrics_all)
    coefficients = coefficient_table_from_results(metrics_all)

    feature_df.to_csv(output_path / "feature_data.csv", index=False)
    metrics_all.to_csv(output_path / "metrics_all.csv", index=False)
    metrics_inside.to_csv(output_path / "metrics_inside_profile.csv", index=False)
    metrics_outside.to_csv(output_path / "metrics_outside_profile.csv", index=False)
    incremental.to_csv(output_path / "incremental_value.csv", index=False)
    coefficients.to_csv(output_path / "coefficients.csv", index=False)

    train_df, test_df = train_test_split_time(feature_df, train_ratio=train_ratio)
    final_model = fit_linear_model(train_df, ["F_liq", "p", "u"], target_col="F_next")
    test_clean = test_df.dropna(subset=["F_liq", "p", "u", "F_next"]).copy()
    pred = predict_model(final_model, test_clean, ["F_liq", "p", "u"])
    residuals = test_clean["F_next"].to_numpy(dtype=float) - pred

    plot_kinetic_activity(feature_df, figures_path / "kinetic_activity.png")
    plot_momentum(feature_df, figures_path / "momentum.png")
    plot_force_scatter(feature_df, figures_path / "force_scatter.png")
    plot_actual_vs_predicted(test_clean, pred, figures_path / "actual_vs_predicted.png")
    plot_residuals(residuals, figures_path / "residuals.png")
    plot_target_vs_negative_p(feature_df, figures_path / "target_vs_negative_p.png")
    plot_fliq_residual_added_value(feature_df, figures_path / "fliq_vs_p_residual.png")
    plot_inside_outside_force_scatter(feature_df, figures_path / "force_scatter_inside_outside.png")
    plot_profile_diagnostics(feature_df, figures_path / "profile_diagnostics.png")
    snapshot_idx = min(max(profile_window, len(feature_df) // 2), len(feature_df) - 1)
    plot_volume_profile_snapshot(
        feature_df,
        idx=snapshot_idx,
        profile_window=profile_window,
        bins=bins,
        output_path=figures_path / "volume_profile_snapshot.png",
        mode=potential_mode,
        profile_method=profile_method,
    )

    _write_report(
        reports_path / "report.md",
        feature_df=feature_df,
        metrics_all=metrics_all,
        metrics_inside=metrics_inside,
        metrics_outside=metrics_outside,
        incremental=incremental,
        volume_window=volume_window,
        profile_window=profile_window,
        bins=bins,
        potential_mode=potential_mode,
        profile_method=profile_method,
    )

    print("Lagrangian Market Mechanics v0.2 experiment complete")
    print(f"Rows: {len(feature_df)}")
    print(f"Outside profile range: {feature_df['outside_profile_range'].mean() * 100.0:.2f}%")
    print(f"Output directory: {output_path}")
    print("Metrics:")
    print(metrics_all.to_string(index=False))
    print("Incremental liquidity value:")
    print(incremental.to_string(index=False))
    return metrics_all


def run_experiment(
    csv_path: str,
    output_dir: str = "outputs",
    volume_window: int = 500,
    profile_window: int = 500,
    bins: int = 50,
    potential_mode: str = "barrier",
    profile_method: str = "close",
    train_ratio: float = 0.7,
    compare_modes: bool = False,
    min_samples: int = 100,
) -> pd.DataFrame:
    """Run the full v0.2 research workflow and save experiment artifacts."""
    if compare_modes:
        output_path = Path(output_dir)
        summaries = []
        for mode in ["barrier", "well"]:
            metrics = _run_single_experiment(
                csv_path=csv_path,
                output_dir=output_path / mode,
                volume_window=volume_window,
                profile_window=profile_window,
                bins=bins,
                potential_mode=mode,
                profile_method=profile_method,
                train_ratio=train_ratio,
                min_samples=min_samples,
            )
            best = metrics.sort_values("test_r2", ascending=False).iloc[0].to_dict()
            best["mode"] = mode
            summaries.append(best)
        comparison = pd.DataFrame(summaries)
        comparison.to_csv(output_path / "mode_comparison.csv", index=False)
        print(f"Mode comparison saved to {output_path / 'mode_comparison.csv'}")
        return comparison

    return _run_single_experiment(
        csv_path=csv_path,
        output_dir=output_dir,
        volume_window=volume_window,
        profile_window=profile_window,
        bins=bins,
        potential_mode=potential_mode,
        profile_method=profile_method,
        train_ratio=train_ratio,
        min_samples=min_samples,
    )
