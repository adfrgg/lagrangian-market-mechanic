import pandas as pd
import pytest

from lagrangian_market.evaluation import compute_incremental_value


def test_incremental_value_delta_calculations() -> None:
    results = pd.DataFrame(
        [
            {"model": "baseline_zero", "test_r2": 0.0, "rmse": 2.0, "mae": 1.0, "sign_accuracy_raw": 0.50},
            {"model": "lagrangian_liq_only", "test_r2": 0.1, "rmse": 1.8, "mae": 0.9, "sign_accuracy_raw": 0.55},
            {"model": "baseline_u_only", "test_r2": 0.2, "rmse": 1.5, "mae": 0.8, "sign_accuracy_raw": 0.60},
            {"model": "lagrangian_liq_u", "test_r2": 0.15, "rmse": 1.6, "mae": 0.85, "sign_accuracy_raw": 0.61},
            {"model": "baseline_p_only", "test_r2": 0.3, "rmse": 1.2, "mae": 0.7, "sign_accuracy_raw": 0.62},
            {"model": "lagrangian_liq_p", "test_r2": 0.31, "rmse": 1.1, "mae": 0.69, "sign_accuracy_raw": 0.63},
            {"model": "baseline_p_u", "test_r2": 0.4, "rmse": 1.0, "mae": 0.6, "sign_accuracy_raw": 0.65},
            {"model": "lagrangian_liq_p_u", "test_r2": 0.39, "rmse": 1.05, "mae": 0.58, "sign_accuracy_raw": 0.66},
        ]
    )

    out = compute_incremental_value(results)
    liq_only = out.loc[out["comparison"] == "liq_only_vs_zero"].iloc[0]
    liq_u = out.loc[out["comparison"] == "liq_u_vs_u"].iloc[0]

    assert liq_only["delta_test_R2"] == 0.1
    assert liq_only["delta_RMSE"] == pytest.approx(0.2)
    assert liq_only["conclusion"] == "positive"
    assert liq_u["conclusion"] == "weak"
