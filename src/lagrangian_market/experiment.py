"""End-to-end experiment runner for Lagrangian Market Mechanics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lagrangian_market.data import load_ohlcv
from lagrangian_market.evaluation import run_model_suite
from lagrangian_market.liquidity import add_liquidity_force
from lagrangian_market.mechanics import add_mechanics_features
from lagrangian_market.models import fit_linear_model, predict_model, train_test_split_time
from lagrangian_market.plots import (
    plot_actual_vs_predicted,
    plot_force_scatter,
    plot_kinetic_activity,
    plot_momentum,
    plot_residuals,
    plot_volume_profile_snapshot,
)


def _format_date_range(df: pd.DataFrame) -> str:
    if "timestamp" not in df.columns or df.empty:
        return "unknown"
    return f"{df['timestamp'].min()} to {df['timestamp'].max()}"


def _write_report(
    report_path: Path,
    feature_df: pd.DataFrame,
    metrics: pd.DataFrame,
    volume_window: int,
    profile_window: int,
    bins: int,
    potential_mode: str,
) -> None:
    best = metrics.sort_values("test_r2", ascending=False).iloc[0]
    lag_model = metrics.loc[metrics["model"] == "lagrangian_liq_damping"]
    lag_row = lag_model.iloc[0] if not lag_model.empty else best

    report = f"""# Lagrangian Market Mechanics v0.1 Report

## Dataset

- Rows after feature construction: {len(feature_df)}
- Date range: {_format_date_range(feature_df)}
- Volume normalization window: {volume_window}
- Liquidity profile window: {profile_window}
- Histogram bins: {bins}
- Potential mode: {potential_mode}

## Best Model By Test R2

- Model: {best['model']}
- Test R2: {best['test_r2']:.6g}
- Sign accuracy: {best['sign_accuracy']:.4f}
- Test RMSE: {best['rmse']:.6g}
- Test MAE: {best['mae']:.6g}

## Lagrangian Liquidity + Damping Model

- Coefficient of F_liq: {lag_row['coef_F_liq']:.6g}
- Coefficient of u: {lag_row['coef_u']:.6g}
- Test R2: {lag_row['test_r2']:.6g}
- Sign accuracy: {lag_row['sign_accuracy']:.4f}

## Interpretation Notes

This experiment tests whether a rolling volume-profile potential contains
information about the next observed change in volume-weighted price momentum.
It is not a trading strategy and does not claim markets obey physical laws.
Positive evidence would appear as stable out-of-sample explanatory power,
directional accuracy above naive baselines, and interpretable coefficients.
Weak or unstable results suggest the OHLCV-derived potential is too crude, the
force estimate is too noisy, or the asset/regime does not support this
approximation.
"""
    report_path.write_text(report, encoding="utf-8")


def run_experiment(
    csv_path: str,
    output_dir: str = "outputs",
    volume_window: int = 500,
    profile_window: int = 500,
    bins: int = 50,
    potential_mode: str = "barrier",
    train_ratio: float = 0.7,
) -> pd.DataFrame:
    """Run the full research workflow and save experiment artifacts."""
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
    )
    feature_df = df.dropna(subset=["x", "u", "m", "p", "K", "F_obs", "F_next", "F_liq"]).copy()
    if len(feature_df) < 10:
        raise ValueError("Not enough valid rows after feature construction")

    metrics = run_model_suite(feature_df, train_ratio=train_ratio)
    metrics.to_csv(output_path / "metrics.csv", index=False)
    feature_df.to_csv(output_path / "feature_data.csv", index=False)

    train_df, test_df = train_test_split_time(feature_df, train_ratio=train_ratio)
    final_model = fit_linear_model(train_df, ["F_liq", "u"], target_col="F_next")
    test_clean = test_df.dropna(subset=["F_liq", "u", "F_next"]).copy()
    pred = predict_model(final_model, test_clean, ["F_liq", "u"])
    residuals = test_clean["F_next"].to_numpy(dtype=float) - pred

    plot_kinetic_activity(feature_df, figures_path / "kinetic_activity.png")
    plot_momentum(feature_df, figures_path / "momentum.png")
    plot_force_scatter(feature_df, figures_path / "force_scatter.png")
    plot_actual_vs_predicted(test_clean, pred, figures_path / "actual_vs_predicted.png")
    plot_residuals(residuals, figures_path / "residuals.png")
    snapshot_idx = min(max(profile_window, len(feature_df) // 2), len(feature_df) - 1)
    plot_volume_profile_snapshot(
        feature_df,
        idx=snapshot_idx,
        profile_window=profile_window,
        bins=bins,
        output_path=figures_path / "volume_profile_snapshot.png",
        mode=potential_mode,
    )

    _write_report(
        reports_path / "report.md",
        feature_df=feature_df,
        metrics=metrics,
        volume_window=volume_window,
        profile_window=profile_window,
        bins=bins,
        potential_mode=potential_mode,
    )

    print("Lagrangian Market Mechanics experiment complete")
    print(f"Rows: {len(feature_df)}")
    print(f"Date range: {_format_date_range(feature_df)}")
    print(f"Output directory: {output_path}")
    print("Metrics:")
    print(metrics.to_string(index=False))

    return metrics
