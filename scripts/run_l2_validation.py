"""Run L2 validation across processed symbols and bar sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.l2_experiment import load_l2_dataset  # noqa: E402
from lagrangian_market.l2_models import compute_l2_incremental_value, run_l2_fold_metrics, run_l2_model_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate L2 models across processed datasets.")
    parser.add_argument("--processed-dir", default="data/l2_processed", help="Processed dataset directory.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], help="Symbols.")
    parser.add_argument("--bar-sizes", nargs="+", default=["1s", "5s", "15s", "60s"], help="Bar sizes.")
    parser.add_argument("--output", default="outputs/l2_validation", help="Validation output directory.")
    parser.add_argument("--folds", type=int, default=5, help="Chronological folds.")
    return parser.parse_args()


def _find_dataset(directory: Path, symbol: str, bar_size: str) -> Path | None:
    for suffix in [".parquet", ".csv"]:
        path = directory / f"{symbol}_{bar_size}{suffix}"
        if path.exists():
            return path
    return None


def _robust_positive(row: pd.Series) -> bool:
    return (
        row["comparison"] == "l2_full_vs_p_u"
        and row["delta_test_R2"] > 0
        and row["delta_RMSE"] > 0
        and row.get("fold_positive_rate", 0) >= 0.6
        and row.get("fpot_coef_sign_stability", 0) >= 0.6
        and row.get("sample_size", 0) >= 1000
    )


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for symbol in args.symbols:
        for bar_size in args.bar_sizes:
            path = _find_dataset(processed_dir, symbol.upper(), bar_size)
            if path is None:
                print(f"Skipping missing dataset {symbol}_{bar_size}")
                continue
            print(f"Validating {path}")
            df = load_l2_dataset(path).dropna()
            metrics = run_l2_model_suite(df)
            incremental = compute_l2_incremental_value(metrics)
            _fold_metrics, fold_summary = run_l2_fold_metrics(df, folds=args.folds)
            full_fold = fold_summary.loc[fold_summary["model"] == "l2_full"]
            fold_rate = float(full_fold["positive_test_r2_rate"].iloc[0]) if not full_fold.empty else np.nan
            sign_stability = (
                float(full_fold["fpot_coef_sign_stability"].iloc[0])
                if not full_fold.empty and "fpot_coef_sign_stability" in full_fold.columns
                else np.nan
            )
            for _, inc in incremental.iterrows():
                row = inc.to_dict()
                row["symbol"] = symbol.upper()
                row["bar_size"] = bar_size
                row["data_path"] = str(path)
                row["sample_size"] = len(df)
                row["fold_positive_rate"] = fold_rate
                row["fpot_coef_sign_stability"] = sign_stability
                row["robust_positive"] = _robust_positive(pd.Series(row))
                rows.append(row)

    if not rows:
        raise RuntimeError("No validation datasets found")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "l2_validation_results.csv", index=False)
    summary_symbol = results.groupby(["symbol", "comparison"]).agg(
        n=("delta_test_R2", "count"),
        mean_delta_test_R2=("delta_test_R2", "mean"),
        mean_delta_RMSE=("delta_RMSE", "mean"),
        robust_positive_rate=("robust_positive", "mean"),
    ).reset_index()
    summary_bar = results.groupby(["bar_size", "comparison"]).agg(
        n=("delta_test_R2", "count"),
        mean_delta_test_R2=("delta_test_R2", "mean"),
        mean_delta_RMSE=("delta_RMSE", "mean"),
        robust_positive_rate=("robust_positive", "mean"),
    ).reset_index()
    top = results.sort_values(["robust_positive", "delta_test_R2", "delta_RMSE"], ascending=[False, False, False])

    summary_symbol.to_csv(output_dir / "summary_by_symbol.csv", index=False)
    summary_bar.to_csv(output_dir / "summary_by_bar_size.csv", index=False)
    top.to_csv(output_dir / "top_configs.csv", index=False)
    (output_dir / "l2_validation_report.md").write_text(
        "# L2 Validation Report\n\n"
        "Robust positive requires l2_full to beat baseline_p_u in test R2 and RMSE, "
        "fold positive rate >= 0.6, F_pot coefficient sign stability >= 0.6, "
        "and sample size >= 1000.\n\n"
        f"```text\n{top.head(30).to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    print(f"Saved validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
