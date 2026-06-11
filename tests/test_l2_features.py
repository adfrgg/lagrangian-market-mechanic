import numpy as np
import pandas as pd

from lagrangian_market.l2_features import build_l2_bars


def _depth_rows(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mid = 100.0 + i * 0.01
        row = {
            "timestamp_event": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i),
            "timestamp_local": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i),
            "symbol": "ETHUSDT",
            "last_update_id": i,
        }
        for level in range(1, 21):
            row[f"bid_price_{level}"] = mid - 0.01 * level
            row[f"bid_qty_{level}"] = 1.0 + level * 0.1
            row[f"ask_price_{level}"] = mid + 0.01 * level
            row[f"ask_qty_{level}"] = 1.2 + level * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def _trade_rows(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp_event": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=500 * i),
                "timestamp_local": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(milliseconds=500 * i),
                "symbol": "ETHUSDT",
                "agg_trade_id": i,
                "price": 100.0,
                "qty": 0.5,
                "quote_qty": 50.0,
                "is_buyer_maker": i % 2 == 0,
            }
        )
    return pd.DataFrame(rows)


def test_build_l2_bars_computes_forces_and_mechanics() -> None:
    bars = build_l2_bars(_depth_rows(), _trade_rows(), "ETHUSDT", bar_size="5s", mass_window=3)

    for col in ["F_pot_5bps", "F_flow", "M", "p", "K", "F_obs", "F_next"]:
        assert col in bars.columns
    assert bars["F_pot_5bps"].notna().any()
    assert bars["F_flow"].notna().any()
    assert bars["M"].notna().any()
