"""Model-suite evaluation for Lagrangian Market Mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lagrangian_market.models import (
    MajoritySignModel,
    NegativePFixedModel,
    ZeroModel,
    evaluate_model,
    fit_linear_model,
    train_test_split_time,
)


def _majority_sign_model(train_df: pd.DataFrame, target_col: str) -> MajoritySignModel:
    signs = np.sign(train_df[target_col].dropna().to_numpy(dtype=float))
    nonzero = signs[signs != 0]
    if len(nonzero) == 0:
        sign = 0.0
    else:
        sign = 1.0 if np.sum(nonzero > 0) >= np.sum(nonzero < 0) else -1.0
    magnitude = float(np.nanmean(np.abs(train_df[target_col].to_numpy(dtype=float))))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        magnitude = 1.0
    return MajoritySignModel(sign=sign, magnitude=magnitude)


def run_model_suite(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    target_col: str = "F_next",
) -> pd.DataFrame:
    """Train and evaluate v0.2 baselines and Lagrangian-inspired models.

    The suite explicitly controls for the algebraic target artifact
    ``F_next_t = p_{t+1} - p_t`` by including fixed ``-p_t`` and fitted ``p_t``
    baselines. All splits are chronological.
    """
    base_cols = ["p", "u", "F_liq", target_col]
    clean = df.dropna(subset=base_cols).copy()
    if len(clean) < 10:
        raise ValueError("Not enough valid rows to run model suite")

    train_df, test_df = train_test_split_time(clean, train_ratio=train_ratio)

    specs: list[tuple[str, list[str], object]] = [
        ("baseline_zero", [], ZeroModel()),
        ("baseline_negative_p_fixed", ["p"], NegativePFixedModel("p")),
        ("majority_sign_baseline", [], _majority_sign_model(train_df, target_col)),
    ]

    linear_specs = [
        ("baseline_p_only", ["p"]),
        ("baseline_u_only", ["u"]),
        ("baseline_p_u", ["p", "u"]),
        ("lagrangian_liq_only", ["F_liq"]),
        ("lagrangian_liq_u", ["F_liq", "u"]),
        ("lagrangian_liq_p", ["F_liq", "p"]),
        ("lagrangian_liq_p_u", ["F_liq", "p", "u"]),
    ]
    for name, features in linear_specs:
        model = fit_linear_model(train_df, features, target_col=target_col)
        specs.append((name, features, model))

    rows = []
    coef_rows = []
    for name, features, model in specs:
        test_metrics = evaluate_model(model, test_df, features, target_col=target_col)
        train_metrics = evaluate_model(model, train_df, features, target_col=target_col)

        row = {
            "model": name,
            "train_r2": train_metrics["r2"],
            "test_r2": test_metrics["r2"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "sign_accuracy": test_metrics["sign_accuracy"],
            "sign_accuracy_raw": test_metrics["sign_accuracy_raw"],
            "sign_accuracy_nonzero_pred": test_metrics["sign_accuracy_nonzero_pred"],
            "corr": test_metrics["corr"],
            "sample_size": int(test_metrics["sample_size"]),
            "coef_F_liq": np.nan,
            "coef_p": np.nan,
            "coef_u": np.nan,
            "intercept": np.nan,
        }

        if hasattr(model, "coef_"):
            coef_map = dict(zip(features, np.asarray(model.coef_, dtype=float)))
            row["coef_F_liq"] = coef_map.get("F_liq", np.nan)
            row["coef_p"] = coef_map.get("p", np.nan)
            row["coef_u"] = coef_map.get("u", np.nan)
            row["intercept"] = float(model.intercept_)
            for feature, coefficient in coef_map.items():
                coef_rows.append(
                    {
                        "model": name,
                        "feature": feature,
                        "coefficient": coefficient,
                        "intercept": float(model.intercept_),
                    }
                )
        elif isinstance(model, NegativePFixedModel):
            row["coef_p"] = -1.0
            row["intercept"] = 0.0
            coef_rows.append({"model": name, "feature": "p", "coefficient": -1.0, "intercept": 0.0})

        rows.append(row)

    results = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    results.attrs["coefficients"] = pd.DataFrame(coef_rows)
    return results


def coefficient_table_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the coefficient table attached by ``run_model_suite`` if present."""
    coefficients = results_df.attrs.get("coefficients")
    if isinstance(coefficients, pd.DataFrame):
        return coefficients.copy()
    return pd.DataFrame(columns=["model", "feature", "coefficient", "intercept"])


def compute_incremental_value(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare liquidity models against matched artifact-control baselines.

    Positive ``delta_rmse`` and ``delta_mae`` mean the liquidity model reduces
    error relative to the baseline.
    """
    comparisons = [
        ("liq_only_vs_zero", "lagrangian_liq_only", "baseline_zero"),
        ("liq_u_vs_u", "lagrangian_liq_u", "baseline_u_only"),
        ("liq_p_vs_p", "lagrangian_liq_p", "baseline_p_only"),
        ("liq_p_u_vs_p_u", "lagrangian_liq_p_u", "baseline_p_u"),
    ]
    by_model = results_df.set_index("model")
    rows = []
    for name, liquidity_model, baseline_model in comparisons:
        if liquidity_model not in by_model.index or baseline_model not in by_model.index:
            continue
        liq = by_model.loc[liquidity_model]
        base = by_model.loc[baseline_model]
        delta_test_r2 = float(liq["test_r2"] - base["test_r2"])
        delta_rmse = float(base["rmse"] - liq["rmse"])
        delta_mae = float(base["mae"] - liq["mae"])
        delta_sign_accuracy = float(liq["sign_accuracy_raw"] - base["sign_accuracy_raw"])

        positives = sum([delta_test_r2 > 0, delta_rmse > 0, delta_mae > 0, delta_sign_accuracy > 0])
        if delta_test_r2 > 0 and delta_rmse > 0:
            conclusion = "positive"
        elif positives > 0:
            conclusion = "weak"
        else:
            conclusion = "negative"

        rows.append(
            {
                "comparison": name,
                "liquidity_model": liquidity_model,
                "baseline_model": baseline_model,
                "delta_test_R2": delta_test_r2,
                "delta_RMSE": delta_rmse,
                "delta_MAE": delta_mae,
                "delta_sign_accuracy": delta_sign_accuracy,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)
