import numpy as np
import pandas as pd

from lagrangian_market.liquidity import add_liquidity_force


def test_outside_profile_range_and_distance() -> None:
    close = np.array([100, 101, 102, 101.5, 104, 120], dtype=float)
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "volume": np.full(len(close), 100.0),
        }
    )
    df["x"] = np.log(df["close"])

    out = add_liquidity_force(df, profile_window=5, bins=10)

    assert bool(out.loc[5, "outside_profile_range"])
    assert out.loc[5, "distance_to_profile_range"] > 0
    assert out.loc[3, "distance_to_profile_range"] == 0
    assert not bool(out.loc[3, "outside_profile_range"])
