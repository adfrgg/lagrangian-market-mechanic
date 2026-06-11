"""Download Binance public OHLCV candles into the project CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.data import download_binance_ohlcv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance spot OHLCV candles.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance spot symbol, for example BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Kline interval, for example 5m, 15m, 1h, 1d.")
    parser.add_argument("--limit", type=int, default=2000, help="Total candles to download.")
    parser.add_argument("--output", default="data/raw/btcusdt_1h.csv", help="Output CSV path.")
    parser.add_argument("--start", default=None, help="Optional UTC start date, for example 2024-01-01.")
    parser.add_argument("--end", default=None, help="Optional UTC end date, for example 2024-06-01.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = download_binance_ohlcv(
        symbol=args.symbol,
        interval=args.interval,
        output_path=args.output,
        limit=args.limit,
        start_time=args.start,
        end_time=args.end,
    )
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
