import numpy as np
import pandas as pd

from lagrangian_market.l2_features import build_l2_bars


def _depth(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mid = 100.0 + i * 0.005
        row = {"timestamp_event": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i), "symbol": "BTCUSDT"}
        for level in range(1, 21):
            row[f"bid_price_{level}"] = mid - 0.01 * level
            row[f"bid_qty_{level}"] = 1.0 + level * 0.05
            row[f"ask_price_{level}"] = mid + 0.01 * level
            row[f"ask_qty_{level}"] = 1.0 + level * 0.05
        rows.append(row)
    return pd.DataFrame(rows)


def _trades(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_event": [pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=500 * i) for i in range(n)],
            "price": np.full(n, 100.0),
            "qty": np.full(n, 0.2),
            "quote_qty": np.full(n, 20.0),
            "is_buyer_maker": [i % 3 == 0 for i in range(n)],
        }
    )


def test_l2_features_do_not_use_future_rows() -> None:
    idx = 4
    base = build_l2_bars(_depth(), _trades(), "BTCUSDT", bar_size="5s", mass_window=3)
    original = base.loc[idx, ["F_pot_5bps", "F_flow", "M", "p", "K"]].copy()

    changed_depth = _depth()
    changed_trades = _trades()
    cutoff = base.loc[idx, "timestamp"] + pd.Timedelta(seconds=5)
    future_depth = pd.to_datetime(changed_depth["timestamp_event"], utc=True) >= cutoff
    future_trades = pd.to_datetime(changed_trades["timestamp_event"], utc=True) >= cutoff
    for level in range(1, 21):
        changed_depth.loc[future_depth, f"bid_qty_{level}"] *= 100.0
        changed_depth.loc[future_depth, f"ask_qty_{level}"] *= 0.01
    changed_trades.loc[future_trades, "quote_qty"] *= 50.0

    recomputed = build_l2_bars(changed_depth, changed_trades, "BTCUSDT", bar_size="5s", mass_window=3)
    updated = recomputed.loc[idx, ["F_pot_5bps", "F_flow", "M", "p", "K"]]

    assert np.allclose(original.astype(float), updated.astype(float), equal_nan=True)
