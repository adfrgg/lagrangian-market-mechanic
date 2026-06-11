import numpy as np
import pandas as pd

from lagrangian_market.liquidity import add_liquidity_force, build_volume_profile


def test_close_method_preserves_total_volume() -> None:
    x = np.log(np.array([100.0, 101.0, 102.0]))
    volume = np.array([10.0, 20.0, 30.0])

    hist, centers = build_volume_profile(x, volume, bins=5, method="close")

    assert len(hist) == 5
    assert len(centers) == 5
    assert np.isclose(hist.sum(), volume.sum())


def test_typical_method_uses_hlc3() -> None:
    close = np.array([100.0, 102.0, 104.0])
    high = close + 3.0
    low = close - 3.0
    x = np.log(close)
    volume = np.array([10.0, 20.0, 30.0])

    hist_typical, _ = build_volume_profile(
        x,
        volume,
        bins=5,
        method="typical",
        high_hist=high,
        low_hist=low,
        close_hist=close,
    )

    assert np.isclose(hist_typical.sum(), volume.sum())


def test_range_uniform_preserves_total_volume() -> None:
    close = np.array([100.0, 102.0, 104.0])
    high = close + 2.0
    low = close - 2.0
    x = np.log(close)
    volume = np.array([10.0, 20.0, 30.0])

    hist, _ = build_volume_profile(
        x,
        volume,
        bins=8,
        method="range_uniform",
        high_hist=high,
        low_hist=low,
        close_hist=close,
    )

    assert np.isclose(hist.sum(), volume.sum())


def test_range_uniform_uses_no_future_data() -> None:
    close = np.linspace(100.0, 110.0, 30)
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(100.0, 200.0, 30),
        }
    )
    df["x"] = np.log(df["close"])
    base = add_liquidity_force(df, profile_window=10, bins=8, profile_method="range_uniform")
    original = base.loc[15, "F_liq"]

    changed = df.copy()
    changed.loc[16:, ["close", "high", "low", "volume"]] = changed.loc[16:, ["close", "high", "low", "volume"]] * 10.0
    changed["x"] = np.log(changed["close"])
    recomputed = add_liquidity_force(changed, profile_window=10, bins=8, profile_method="range_uniform")

    assert np.isclose(original, recomputed.loc[15, "F_liq"], equal_nan=True)
