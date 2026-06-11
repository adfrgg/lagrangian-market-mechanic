import pandas as pd

from lagrangian_market.l2_features import aggregate_trades_to_bars, build_l2_bars


def _depth_rows(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        mid = 100.0 + i * 0.01
        row = {
            "timestamp_event": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(seconds=i),
            "symbol": "ETHUSDT",
            "last_update_id": i,
        }
        for level in range(1, 21):
            row[f"bid_price_{level}"] = mid - 0.01 * level
            row[f"bid_qty_{level}"] = 1.0
            row[f"ask_price_{level}"] = mid + 0.01 * level
            row[f"ask_qty_{level}"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_trade_aggregation_buy_sell_classification() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp_event": pd.Timestamp("2024-01-01 00:00:00.100", tz="UTC"),
                "price": 100.0,
                "qty": 1.0,
                "quote_qty": 100.0,
                "is_buyer_maker": False,
            },
            {
                "timestamp_event": pd.Timestamp("2024-01-01 00:00:00.200", tz="UTC"),
                "price": 100.0,
                "qty": 2.0,
                "quote_qty": 200.0,
                "is_buyer_maker": True,
            },
        ]
    )

    bars = aggregate_trades_to_bars(trades, "1s")

    assert bars.loc[0, "taker_buy_quote"] == 100.0
    assert bars.loc[0, "taker_sell_quote"] == 200.0
    assert bars.loc[0, "num_trades"] == 2


def test_depth_alignment_uses_last_snapshot_inside_bar() -> None:
    depth = _depth_rows(8)
    trades = pd.DataFrame(columns=["timestamp_event", "price", "qty", "quote_qty", "is_buyer_maker"])

    bars = build_l2_bars(depth, trades, "ETHUSDT", bar_size="5s", mass_window=2)

    first_mid_expected = 100.0 + 4 * 0.01
    assert abs(bars.loc[0, "mid"] - first_mid_expected) < 1e-9
