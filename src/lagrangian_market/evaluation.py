"""Model-suite evaluation for Lagrangian Market Mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lagrangian_market.models import (
    ColumnPersistenceModel,
    ZeroModel,
    evaluate_model,
    fit_linear_model,
    train_test_split_time,
)


def run_model_suite(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    target_col: str = "F_next",
) -> pd.DataFrame:
    """Train and evaluate baseline and Lagrangian-inspired models.

    All splits are chronological. No random shuffle is used.
    """
    base_cols = ["F_obs", "u", "F_liq", target_col]
    clean = df.dropna(subset=base_cols).copy()
    if len(clean) < 10:
        raise ValueError("Not enough valid rows to run model suite")

    train_df, test_df = train_test_split_time(clean, train_ratio=train_ratio)

    specs: list[tuple[str, list[str], object]] = [
        ("baseline_zero", [], ZeroModel()),
        ("baseline_momentum", ["F_obs"], ColumnPersistenceModel("F_obs")),
    ]

    fitted_specs: list[tuple[str, list[str], object]] = []
    linear_specs = [
        ("baseline_velocity", ["u"]),
        ("lagrangian_liq_only", ["F_liq"]),
        ("lagrangian_liq_damping", ["F_liq", "u"]),
    ]
    for name, features in linear_specs:
        model = fit_linear_model(train_df, features, target_col=target_col)
        fitted_specs.append((name, features, model))

    rows = []
    for name, features, model in specs + fitted_specs:
        test_metrics = evaluate_model(model, test_df, features, target_col=target_col)
        train_metrics = evaluate_model(model, train_df, features, target_col=target_col)

        row = {
            "model": name,
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "sign_accuracy": test_metrics["sign_accuracy"],
            "corr": test_metrics["corr"],
            "dtw_raw_norm": test_metrics["dtw_raw_norm"],
            "dtw_z_norm": test_metrics["dtw_z_norm"],
            "sample_size": int(test_metrics["sample_size"]),
            "coef_F_liq": np.nan,
            "coef_u": np.nan,
            "intercept": np.nan,
        }

        if hasattr(model, "coef_"):
            coef_map = dict(zip(features, np.asarray(model.coef_, dtype=float)))
            row["coef_F_liq"] = coef_map.get("F_liq", np.nan)
            row["coef_u"] = coef_map.get("u", np.nan)
            row["intercept"] = float(model.intercept_)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
