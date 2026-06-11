"""Run a compact parameter grid and save a metrics summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.data import load_ohlcv  # noqa: E402
from lagrangian_market.evaluation import run_model_suite  # noqa: E402
from lagrangian_market.features import build_feature_set  # noqa: E402


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LMM parameter variations.")
    parser.add_argument("--csv", required=True, help="Input OHLCV CSV.")
    parser.add_argument("--output", default="outputs/variation_grid_summary.csv", help="Summary CSV path.")
    parser.add_argument("--windows", default="50,100,200", help="Comma-separated volume/profile windows.")
    parser.add_argument("--bins", default="25,50,100", help="Comma-separated histogram bin counts.")
    parser.add_argument("--modes", default="barrier,well", help="Comma-separated modes: barrier,well.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = _parse_ints(args.windows)
    bins_values = _parse_ints(args.bins)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    raw = load_ohlcv(args.csv)
    rows = []

    for mode in modes:
        for window in windows:
            for bins in bins_values:
                print(f"Running mode={mode}, window={window}, bins={bins}")
                features = build_feature_set(
                    raw,
                    volume_window=window,
                    profile_window=window,
                    bins=bins,
                    potential_mode=mode,
                )
                metrics = run_model_suite(features, train_ratio=args.train_ratio)
                for _, metric in metrics.iterrows():
                    row = metric.to_dict()
                    row["mode"] = mode
                    row["window"] = window
                    row["bins"] = bins
                    rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_cols = ["mode", "window", "bins"] + [
        col for col in summary.columns if col not in {"mode", "window", "bins"}
    ]
    summary = summary[ordered_cols]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    best = summary.sort_values("test_r2", ascending=False).head(10)
    print("\nTop 10 rows by test_r2:")
    print(
        best[
            [
                "mode",
                "window",
                "bins",
                "model",
                "test_r2",
                "sign_accuracy",
                "corr",
                "coef_F_liq",
                "coef_u",
                "sample_size",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary to {output}")


if __name__ == "__main__":
    main()
