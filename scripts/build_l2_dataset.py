"""Build fixed-time L2 mechanics bars from raw Binance depth/trade files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.l2_features import build_l2_dataset_from_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed L2 dataset from raw depth/trade data.")
    parser.add_argument("--input", required=True, help="Raw L2 input directory.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. ETHUSDT.")
    parser.add_argument("--bar-size", default="5s", help="Bar size, e.g. 1s, 5s, 15s, 60s.")
    parser.add_argument("--depth-levels", type=int, default=20, help="Depth levels in raw snapshots.")
    parser.add_argument("--horizon-bps", nargs="+", type=int, default=[1, 2, 5, 10], help="Potential horizons in bps.")
    parser.add_argument("--mass-window", type=int, default=300, help="Rolling mass/flow normalization window.")
    parser.add_argument("--output", required=True, help="Processed output path, parquet or csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_l2_dataset_from_files(
        input_dir=args.input,
        symbol=args.symbol.upper(),
        bar_size=args.bar_size,
        output_path=args.output,
        depth_levels=args.depth_levels,
        horizon_bps=args.horizon_bps,
        mass_window=args.mass_window,
    )
    print(f"Saved processed L2 dataset to {written}")


if __name__ == "__main__":
    main()
