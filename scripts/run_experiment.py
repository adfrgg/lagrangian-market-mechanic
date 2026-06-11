"""Command-line entry point for running a Lagrangian Market Mechanics experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.experiment import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lagrangian Market Mechanics research experiment.")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument("--output", default="outputs/run_001", help="Output directory for artifacts.")
    parser.add_argument("--volume-window", type=int, default=500, help="Rolling volume normalization window.")
    parser.add_argument("--profile-window", type=int, default=500, help="Rolling volume-profile lookback window.")
    parser.add_argument("--bins", type=int, default=50, help="Number of volume-profile histogram bins.")
    parser.add_argument("--mode", choices=["barrier", "well"], default="barrier", help="Potential interpretation.")
    parser.add_argument(
        "--profile-method",
        choices=["close", "typical", "hlc3", "range_uniform", "range_triangular"],
        default="close",
        help="Volume-profile construction method.",
    )
    parser.add_argument("--compare-modes", action="store_true", help="Run both barrier and well modes.")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum sample size for metric tables.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological training fraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        csv_path=args.csv,
        output_dir=args.output,
        volume_window=args.volume_window,
        profile_window=args.profile_window,
        bins=args.bins,
        potential_mode=args.mode,
        profile_method=args.profile_method,
        train_ratio=args.train_ratio,
        compare_modes=args.compare_modes,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
