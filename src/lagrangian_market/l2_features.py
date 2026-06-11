"""Feature construction for order-book Lagrangian Market Mechanics."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from lagrangian_market.orderbook import OrderBook


DEFAULT_HORIZON_BPS = (1, 2, 5, 10)
DEFAULT_DEPTH_LEVELS = (5, 10, 20)
EPS = 1e-12


def read_table(path: str | Path) -> pd.DataFrame:
    """Read parquet when possible, otherwise CSV."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet when possible, falling back to CSV if parquet is unavailable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError, ModuleNotFoundError):
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(path, index=False)
    return path


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp_event" not in df.columns:
        raise ValueError("DataFrame must contain timestamp_event")
    ts = pd.to_datetime(df["timestamp_event"], utc=True, errors="coerce")
    return ts


def orderbook_from_depth_row(row: pd.Series, depth_levels: int = 20) -> OrderBook:
    """Build an OrderBook from one flattened depth snapshot row."""
    bids = []
    asks = []
    for i in range(1, depth_levels + 1):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_qty = row.get(f"bid_qty_{i}", np.nan)
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_qty = row.get(f"ask_qty_{i}", np.nan)
        if pd.notna(bid_price) and pd.notna(bid_qty) and float(bid_qty) > 0:
            bids.append((float(bid_price), float(bid_qty)))
        if pd.notna(ask_price) and pd.notna(ask_qty) and float(ask_qty) > 0:
            asks.append((float(ask_price), float(ask_qty)))
    return OrderBook.from_snapshot(bids, asks)


def compute_book_features_for_row(
    row: pd.Series,
    depth_levels: int = 20,
    horizon_bps: Sequence[int] = DEFAULT_HORIZON_BPS,
) -> dict[str, float]:
    """Compute mid, depth, imbalance, and potential features from one book row."""
    book = orderbook_from_depth_row(row, depth_levels=depth_levels)
    mid = book.mid_price()
    spread = book.spread()
    features: dict[str, float] = {
        "best_bid": book.best_bid(),
        "best_ask": book.best_ask(),
        "mid": mid,
        "log_mid": np.log(mid) if np.isfinite(mid) and mid > 0 else np.nan,
        "spread": spread,
        "spread_bps": (spread / mid * 10000.0) if np.isfinite(spread) and np.isfinite(mid) and mid > 0 else np.nan,
    }
    for n in DEFAULT_DEPTH_LEVELS:
        if n <= depth_levels:
            features[f"bid_depth_{n}"] = book.bid_depth_quote(n)
            features[f"ask_depth_{n}"] = book.ask_depth_quote(n)
            features[f"book_imbalance_{n}"] = book.book_imbalance(n)

    for h in horizon_bps:
        up = book.cost_to_move_up_bps(h)
        down = book.cost_to_move_down_bps(h)
        features[f"U_up_{h}bps"] = up
        features[f"U_down_{h}bps"] = down
        features[f"F_pot_{h}bps"] = (
            np.log(down + EPS) - np.log(up + EPS) if np.isfinite(up) and np.isfinite(down) else np.nan
        )
        features[f"insufficient_depth_up_{h}bps"] = not np.isfinite(up)
        features[f"insufficient_depth_down_{h}bps"] = not np.isfinite(down)
    return features


def aggregate_trades_to_bars(trades_df: pd.DataFrame, bar_size: str) -> pd.DataFrame:
    """Aggregate Binance aggTrade rows into signed aggressive-flow bars."""
    if trades_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "taker_buy_quote", "taker_sell_quote", "total_trade_quote", "trade_imbalance", "num_trades"]
        )
    trades = trades_df.copy()
    trades["timestamp_event"] = _timestamp_series(trades)
    trades = trades.dropna(subset=["timestamp_event"])
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce")
    if "quote_qty" not in trades.columns:
        trades["quote_qty"] = trades["price"] * trades["qty"]
    trades["quote_qty"] = pd.to_numeric(trades["quote_qty"], errors="coerce")
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
    trades["bar_time"] = trades["timestamp_event"].dt.floor(bar_size)
    trades["buy_quote"] = np.where(~trades["is_buyer_maker"], trades["quote_qty"], 0.0)
    trades["sell_quote"] = np.where(trades["is_buyer_maker"], trades["quote_qty"], 0.0)

    agg = trades.groupby("bar_time", as_index=False).agg(
        taker_buy_quote=("buy_quote", "sum"),
        taker_sell_quote=("sell_quote", "sum"),
        total_trade_quote=("quote_qty", "sum"),
        num_trades=("quote_qty", "size"),
    )
    agg = agg.rename(columns={"bar_time": "timestamp"})
    agg["trade_imbalance"] = (
        (agg["taker_buy_quote"] - agg["taker_sell_quote"])
        / (agg["taker_buy_quote"] + agg["taker_sell_quote"] + EPS)
    )
    return agg


def build_l2_bars(
    depth_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    symbol: str,
    bar_size: str = "5s",
    depth_levels: int = 20,
    horizon_bps: Sequence[int] = DEFAULT_HORIZON_BPS,
    mass_window: int = 300,
) -> pd.DataFrame:
    """Convert raw depth snapshots and trades into fixed-time L2 mechanics bars."""
    if mass_window <= 0:
        raise ValueError("mass_window must be positive")
    depth = depth_df.copy()
    depth["timestamp_event"] = _timestamp_series(depth)
    depth = depth.dropna(subset=["timestamp_event"]).sort_values("timestamp_event")
    depth["bar_time"] = depth["timestamp_event"].dt.floor(bar_size)
    depth_last = depth.groupby("bar_time", as_index=False).last().rename(columns={"bar_time": "timestamp"})

    rows = []
    for _, row in depth_last.iterrows():
        features = compute_book_features_for_row(row, depth_levels=depth_levels, horizon_bps=horizon_bps)
        features["timestamp"] = row["timestamp"]
        features["symbol"] = symbol
        rows.append(features)
    bars = pd.DataFrame(rows)
    if bars.empty:
        raise ValueError("No depth bars could be built")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    trade_bars = aggregate_trades_to_bars(trades_df, bar_size)
    bars = bars.merge(trade_bars, on="timestamp", how="left")
    for col in ["taker_buy_quote", "taker_sell_quote", "total_trade_quote", "num_trades"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce").fillna(0.0)
    bars["trade_imbalance"] = pd.to_numeric(bars["trade_imbalance"], errors="coerce").fillna(0.0)

    bars = add_l2_mechanics_features(bars, mass_window=mass_window)
    return bars


def add_l2_mechanics_features(df: pd.DataFrame, mass_window: int = 300) -> pd.DataFrame:
    """Add L2 inertia, momentum, kinetic activity, target, and flow force."""
    out = df.copy()
    out["u"] = out["log_mid"].diff()
    out["a"] = out["u"].diff()
    out["M_depth_20"] = 0.5 * (out["bid_depth_20"] + out["ask_depth_20"])
    depth_mean = out["M_depth_20"].rolling(mass_window, min_periods=mass_window).mean()
    out["M"] = out["M_depth_20"] / depth_mean.replace(0.0, np.nan)
    out["p"] = out["M"] * out["u"]
    out["K"] = 0.5 * out["M"] * out["u"] ** 2
    out["F_obs"] = out["p"].diff()
    out["F_next"] = out["F_obs"].shift(-1)
    quote_mean = out["total_trade_quote"].rolling(mass_window, min_periods=mass_window).mean()
    out["F_flow"] = (out["taker_buy_quote"] - out["taker_sell_quote"]) / quote_mean.replace(0.0, np.nan)
    return out


def build_l2_dataset_from_files(
    input_dir: str | Path,
    symbol: str,
    bar_size: str,
    output_path: str | Path,
    depth_levels: int = 20,
    horizon_bps: Sequence[int] = DEFAULT_HORIZON_BPS,
    mass_window: int = 300,
) -> Path:
    """Load raw files, build L2 bars, and write processed dataset."""
    input_dir = Path(input_dir)
    depth_path = _find_existing(input_dir, f"{symbol}_depth")
    trades_path = _find_existing(input_dir, f"{symbol}_trades")
    if depth_path is None:
        raise FileNotFoundError(f"No depth file found for {symbol} in {input_dir}")
    depth = read_table(depth_path)
    trades = read_table(trades_path) if trades_path is not None else pd.DataFrame()
    bars = build_l2_bars(
        depth_df=depth,
        trades_df=trades,
        symbol=symbol,
        bar_size=bar_size,
        depth_levels=depth_levels,
        horizon_bps=horizon_bps,
        mass_window=mass_window,
    )
    return write_table(bars, output_path)


def _find_existing(directory: Path, stem: str) -> Path | None:
    for suffix in [".parquet", ".csv"]:
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return None
