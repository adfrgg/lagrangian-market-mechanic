"""Run one Order-Book Lagrangian Market Mechanics v1.0 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.l2_experiment import run_l2_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L2 Lagrangian Market Mechanics experiment.")
    parser.add_argument("--data", required=True, help="Processed L2 dataset path.")
    parser.add_argument("--output", required=True, help="Experiment output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train ratio.")
    parser.add_argument("--folds", type=int, default=5, help="Expanding chronological folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_l2_experiment(
        data_path=args.data,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
