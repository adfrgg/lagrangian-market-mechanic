"""Collect free Binance public L2-ish depth and aggregate trade data."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.live_collect import collect_binance_partial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance public L2 depth and aggTrade streams.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Symbols to collect.")
    parser.add_argument("--duration-minutes", type=float, default=15, help="Collection duration.")
    parser.add_argument("--mode", choices=["partial", "local"], default="partial", help="Collection mode.")
    parser.add_argument("--depth-levels", type=int, choices=[5, 10, 20], default=20, help="Partial depth levels.")
    parser.add_argument("--speed", choices=["100ms", "1000ms"], default="100ms", help="Depth stream speed.")
    parser.add_argument("--output", default="data/l2_raw/quick_test", help="Output directory.")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output format.")
    parser.add_argument("--flush-every-seconds", type=int, default=10, help="Flush interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "local":
        print("Local book mode is not implemented yet; using partial depth mode.")
    try:
        asyncio.run(
            collect_binance_partial(
                symbols=args.symbols,
                duration_minutes=args.duration_minutes,
                output_dir=args.output,
                depth_levels=args.depth_levels,
                speed=args.speed,
                file_format=args.format,
                flush_every_seconds=args.flush_every_seconds,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted. Buffered data was flushed at the last flush interval.")


if __name__ == "__main__":
    main()
