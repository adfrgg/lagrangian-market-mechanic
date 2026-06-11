"""Async Binance public L2 collector utilities."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


BINANCE_STREAM_URL = "wss://stream.binance.com:9443/stream?streams="


def depth_stream_name(symbol: str, levels: int = 20, speed: str = "100ms") -> str:
    return f"{symbol.lower()}@depth{levels}@{speed}"


def agg_trade_stream_name(symbol: str) -> str:
    return f"{symbol.lower()}@aggTrade"


def parse_partial_depth(symbol: str, payload: dict[str, Any], depth_levels: int = 20) -> dict[str, Any]:
    """Flatten Binance partial depth payload into one row."""
    data = payload.get("data", payload)
    local_ts = pd.Timestamp.utcnow()
    event_ts = pd.to_datetime(data.get("E"), unit="ms", utc=True) if data.get("E") is not None else local_ts
    row: dict[str, Any] = {
        "timestamp_event": event_ts,
        "timestamp_local": local_ts,
        "symbol": data.get("s", symbol).upper(),
        "last_update_id": data.get("lastUpdateId") or data.get("u"),
    }
    bids = data.get("bids") or data.get("b") or []
    asks = data.get("asks") or data.get("a") or []
    for i in range(1, depth_levels + 1):
        bid = bids[i - 1] if i <= len(bids) else [None, None]
        ask = asks[i - 1] if i <= len(asks) else [None, None]
        row[f"bid_price_{i}"] = float(bid[0]) if bid[0] is not None else None
        row[f"bid_qty_{i}"] = float(bid[1]) if bid[1] is not None else None
        row[f"ask_price_{i}"] = float(ask[0]) if ask[0] is not None else None
        row[f"ask_qty_{i}"] = float(ask[1]) if ask[1] is not None else None
    return row


def parse_agg_trade(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten Binance aggTrade payload into one row."""
    data = payload.get("data", payload)
    price = float(data["p"])
    qty = float(data["q"])
    return {
        "timestamp_event": pd.to_datetime(data.get("T") or data.get("E"), unit="ms", utc=True),
        "timestamp_local": pd.Timestamp.utcnow(),
        "symbol": data.get("s", symbol).upper(),
        "agg_trade_id": data.get("a"),
        "price": price,
        "qty": qty,
        "quote_qty": price * qty,
        "is_buyer_maker": bool(data.get("m")),
    }


def flush_symbol_buffers(
    output_dir: str | Path,
    symbol: str,
    depth_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    file_format: str = "parquet",
) -> None:
    """Append buffered rows to per-symbol files and clear buffers."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if depth_rows:
        _append_table(pd.DataFrame(depth_rows), output / f"{symbol}_depth", file_format)
        depth_rows.clear()
    if trade_rows:
        _append_table(pd.DataFrame(trade_rows), output / f"{symbol}_trades", file_format)
        trade_rows.clear()


def _append_table(df: pd.DataFrame, stem: Path, file_format: str) -> None:
    if file_format == "parquet":
        path = stem.with_suffix(".parquet")
        try:
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            pd.concat([existing, df], ignore_index=True).to_parquet(path, index=False)
            return
        except (ImportError, ValueError, ModuleNotFoundError):
            pass
    path = stem.with_suffix(".csv")
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


async def collect_binance_partial(
    symbols: list[str],
    duration_minutes: float,
    output_dir: str | Path,
    depth_levels: int = 20,
    speed: str = "100ms",
    file_format: str = "parquet",
    flush_every_seconds: int = 10,
) -> None:
    """Collect partial depth and aggTrade streams from Binance public WebSocket."""
    try:
        import websockets
    except ImportError as exc:
        raise ImportError("Install websockets to use live collection: pip install websockets") from exc

    symbols = [symbol.upper() for symbol in symbols]
    streams = []
    for symbol in symbols:
        streams.append(depth_stream_name(symbol, levels=depth_levels, speed=speed))
        streams.append(agg_trade_stream_name(symbol))
    url = BINANCE_STREAM_URL + "/".join(streams)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "symbols": symbols,
        "duration_minutes": duration_minutes,
        "mode": "partial",
        "depth_levels": depth_levels,
        "speed": speed,
        "format": file_format,
        "started_utc": pd.Timestamp.utcnow().isoformat(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    buffers = {symbol: {"depth": [], "trades": []} for symbol in symbols}
    end_time = time.monotonic() + duration_minutes * 60.0
    last_flush = time.monotonic()
    last_log = time.monotonic()

    while time.monotonic() < end_time:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
                async for message in ws:
                    payload = json.loads(message)
                    data = payload.get("data", {})
                    event = data.get("e")
                    symbol = str(data.get("s") or str(payload.get("stream", "")).split("@")[0]).upper()
                    if symbol not in buffers:
                        continue
                    if event == "depthUpdate" or "bids" in data:
                        buffers[symbol]["depth"].append(parse_partial_depth(symbol, payload, depth_levels))
                    elif event == "aggTrade":
                        buffers[symbol]["trades"].append(parse_agg_trade(symbol, payload))

                    now = time.monotonic()
                    if now - last_flush >= flush_every_seconds:
                        for sym in symbols:
                            flush_symbol_buffers(output, sym, buffers[sym]["depth"], buffers[sym]["trades"], file_format)
                        last_flush = now
                    if now - last_log >= 10:
                        counts = ", ".join(
                            f"{sym}:d{len(buffers[sym]['depth'])}/t{len(buffers[sym]['trades'])}" for sym in symbols
                        )
                        print(f"collecting {counts}")
                        last_log = now
                    if now >= end_time:
                        break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"WebSocket disconnected: {exc}. Reconnecting in 2s.")
            await asyncio.sleep(2)

    for sym in symbols:
        flush_symbol_buffers(output, sym, buffers[sym]["depth"], buffers[sym]["trades"], file_format)
