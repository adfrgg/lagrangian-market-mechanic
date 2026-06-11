import numpy as np
import pandas as pd

from lagrangian_market.liquidity import add_liquidity_force
from lagrangian_market.mechanics import add_mechanics_features


def _make_df(n: int = 60) -> pd.DataFrame:
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0) * 0.2 + 0.05)
    volume = 1000.0 + 100.0 * np.cos(np.arange(n) / 7.0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
        }
    )


def test_liquidity_force_uses_no_future_data() -> None:
    idx = 30
    base = add_mechanics_features(_make_df(), volume_window=5)
    base = add_liquidity_force(base, profile_window=12, bins=12)
    original_force = base.loc[idx, "F_liq"]

    changed = _make_df()
    changed.loc[idx + 1 :, "close"] = changed.loc[idx + 1 :, "close"] * 3.0
    changed.loc[idx + 1 :, "volume"] = changed.loc[idx + 1 :, "volume"] * 100.0
    changed = add_mechanics_features(changed, volume_window=5)
    changed = add_liquidity_force(changed, profile_window=12, bins=12)
    changed_force = changed.loc[idx, "F_liq"]

    assert np.isclose(original_force, changed_force, equal_nan=True)
