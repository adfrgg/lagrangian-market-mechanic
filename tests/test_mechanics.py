import numpy as np
import pandas as pd

from lagrangian_market.mechanics import add_mechanics_features


def test_mechanics_features_match_definitions() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "open": [100, 101, 103, 102, 104, 106],
            "high": [101, 103, 104, 105, 107, 108],
            "low": [99, 100, 101, 101, 103, 105],
            "close": [100, 102, 101, 105, 104, 108],
            "volume": [10, 20, 30, 40, 50, 60],
        }
    )

    out = add_mechanics_features(df, volume_window=3)

    expected_x = np.log(df["close"])
    expected_u = expected_x.diff()
    expected_m = df["volume"] / df["volume"].rolling(3, min_periods=3).mean()
    expected_p = expected_m * expected_u
    expected_k = 0.5 * expected_m * expected_u**2
    expected_f_obs = expected_p.diff()

    assert np.allclose(out["x"], expected_x, equal_nan=True)
    assert np.allclose(out["u"], expected_u, equal_nan=True)
    assert np.allclose(out["m"], expected_m, equal_nan=True)
    assert np.allclose(out["p"], expected_p, equal_nan=True)
    assert np.allclose(out["K"], expected_k, equal_nan=True)
    assert np.allclose(out["F_obs"], expected_f_obs, equal_nan=True)
    assert np.allclose(out["F_next"], expected_f_obs.shift(-1), equal_nan=True)
