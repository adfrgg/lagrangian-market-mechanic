"""Data loading and free OHLCV download utilities."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load and clean an OHLCV CSV.

    Expected columns are timestamp, open, high, low, close, and volume. Column
    names are lowercased, timestamps are parsed, rows are sorted
    chronologically, invalid close/volume rows are removed, and a clean
    DataFrame is returned.
    """
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    return df.sort_values("timestamp").reset_index(drop=True)


def download_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    output_path: str | Path = "data/raw/btcusdt_1h.csv",
    limit: int = 1000,
    start_time: str | None = None,
    end_time: str | None = None,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    """Download Binance public spot candles and save project-ready OHLCV CSV.

    Binance returns at most 1000 candles per request, so this function paginates
    forward until ``limit`` rows have been collected or the endpoint stops
    returning data.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    symbol = symbol.upper()
    remaining = limit
    rows: list[list[Any]] = []

    start_ms = _to_milliseconds(start_time) if start_time else None
    end_ms = _to_milliseconds(end_time) if end_time else None

    while remaining > 0:
        batch_limit = min(1000, remaining)
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": batch_limit,
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        rows.extend(batch)
        remaining = limit - len(rows)

        last_open_time = int(batch[-1][0])
        next_start_ms = last_open_time + 1
        if start_ms is not None and next_start_ms <= start_ms:
            break
        start_ms = next_start_ms

        if len(batch) < batch_limit:
            break
        if remaining > 0:
            time.sleep(pause_seconds)

    if not rows:
        raise ValueError("No Binance kline rows were returned")

    df = _binance_rows_to_ohlcv(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def _to_milliseconds(value: str) -> int:
    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse timestamp: {value}")
    return int(timestamp.timestamp() * 1000)


def _binance_rows_to_ohlcv(rows: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    raw = pd.DataFrame(rows, columns=columns)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["open_time"], unit="ms", utc=True),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce"),
        }
    )
    return df.dropna(subset=["timestamp", "close", "volume"]).reset_index(drop=True)
