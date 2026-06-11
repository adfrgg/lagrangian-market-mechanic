"""Run a v0.2 robustness sweep for the liquidity-potential hypothesis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lagrangian_market.data import download_binance_ohlcv  # noqa: E402
from lagrangian_market.experiment import run_experiment  # noqa: E402


TARGET_COMPARISONS = ["liq_p_vs_p", "liq_p_u_vs_p_u"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v0.2 robustness sweep across assets, intervals, profile methods, modes, windows, and bins."
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Binance spot symbols.")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m"], help="Binance kline intervals.")
    parser.add_argument("--methods", nargs="+", default=["close", "range_uniform"], help="Profile methods.")
    parser.add_argument("--modes", nargs="+", default=["barrier", "well"], help="Potential modes.")
    parser.add_argument("--profile-windows", nargs="+", type=int, default=[200, 500], help="Profile windows.")
    parser.add_argument("--bins", nargs="+", type=int, default=[80], help="Histogram bin counts.")
    parser.add_argument("--output", default="outputs/sweep", help="Sweep output directory.")
    parser.add_argument("--limit", type=int, default=5000, help="Candles to download per symbol/interval.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing data and experiment outputs.")
    return parser.parse_args()


def _safe_name(*parts: object) -> str:
    return "_".join(str(part).replace("/", "-").replace("\\", "-") for part in parts)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _comparison_delta(metrics: pd.DataFrame | None, comparison: str) -> dict[str, float | str | None]:
    if metrics is None or "model" not in metrics.columns:
        return {
            "delta_test_R2": None,
            "delta_RMSE": None,
            "delta_MAE": None,
            "delta_sign_accuracy": None,
            "sample_size": None,
            "status": "unavailable",
        }

    pairs = {
        "liq_p_vs_p": ("lagrangian_liq_p", "baseline_p_only"),
        "liq_p_u_vs_p_u": ("lagrangian_liq_p_u", "baseline_p_u"),
    }
    liquidity_model, baseline_model = pairs[comparison]
    by_model = metrics.set_index("model")
    if liquidity_model not in by_model.index or baseline_model not in by_model.index:
        return {
            "delta_test_R2": None,
            "delta_RMSE": None,
            "delta_MAE": None,
            "delta_sign_accuracy": None,
            "sample_size": None,
            "status": "missing_models",
        }

    liq = by_model.loc[liquidity_model]
    base = by_model.loc[baseline_model]
    return {
        "delta_test_R2": float(liq["test_r2"] - base["test_r2"]),
        "delta_RMSE": float(base["rmse"] - liq["rmse"]),
        "delta_MAE": float(base["mae"] - liq["mae"]),
        "delta_sign_accuracy": float(liq["sign_accuracy_raw"] - base["sign_accuracy_raw"]),
        "sample_size": int(liq["sample_size"]) if pd.notna(liq["sample_size"]) else None,
        "status": "ok",
    }


def _summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(["comparison", group_col], dropna=False)
    return grouped.agg(
        n=("delta_test_R2", "count"),
        mean_delta_test_R2=("delta_test_R2", "mean"),
        median_delta_test_R2=("delta_test_R2", "median"),
        max_delta_test_R2=("delta_test_R2", "max"),
        mean_delta_RMSE=("delta_RMSE", "mean"),
        mean_delta_MAE=("delta_MAE", "mean"),
        mean_delta_sign_accuracy=("delta_sign_accuracy", "mean"),
        positive_rate=("conclusion", lambda values: float((values == "positive").mean())),
        weak_rate=("conclusion", lambda values: float((values == "weak").mean())),
        negative_rate=("conclusion", lambda values: float((values == "negative").mean())),
    ).reset_index()


def _best_group(summary: pd.DataFrame, group_col: str, comparison: str) -> str:
    subset = summary.loc[summary["comparison"] == comparison].sort_values("mean_delta_test_R2", ascending=False)
    if subset.empty:
        return "not available"
    row = subset.iloc[0]
    return (
        f"{row[group_col]} "
        f"(mean delta_test_R2={row['mean_delta_test_R2']:.6g}, positive_rate={row['positive_rate']:.2f})"
    )


def _write_report(
    output_dir: Path,
    results: pd.DataFrame,
    summary_interval: pd.DataFrame,
    summary_method: pd.DataFrame,
    summary_mode: pd.DataFrame,
    top_configs: pd.DataFrame,
) -> None:
    target = "liq_p_vs_p"
    target_rows = results.loc[results["comparison"] == target]
    positive_rate = float((target_rows["conclusion"] == "positive").mean()) if not target_rows.empty else 0.0
    mean_delta = float(target_rows["delta_test_R2"].mean()) if not target_rows.empty else float("nan")

    consistent_text = (
        "F_liq improves over the p_t baseline consistently enough to justify deeper testing."
        if positive_rate >= 0.6 and mean_delta > 0
        else "F_liq does not improve over the p_t baseline consistently across the sweep."
    )

    outside_available = results.dropna(subset=["outside_delta_test_R2"])
    if outside_available.empty:
        outside_text = "Outside-profile samples were often insufficient, so outside-range performance is inconclusive."
    else:
        inside_mean = results["inside_delta_test_R2"].mean()
        outside_mean = outside_available["outside_delta_test_R2"].mean()
        outside_text = (
            f"Mean inside delta_test_R2 is {inside_mean:.6g}; "
            f"mean outside delta_test_R2 is {outside_mean:.6g}."
        )

    order_book_text = (
        "There is enough positive robustness evidence to consider order-book v0.3."
        if positive_rate >= 0.6 and mean_delta > 0
        else "Evidence is not strong enough by itself; order-book v0.3 is still the right next step because OHLCV profiles are limited."
    )

    report = f"""# v0.2 Robustness Sweep Report

This sweep tests whether `F_liq` adds incremental out-of-sample value beyond
the mechanical `p_t` control implied by `F_next_t = p_(t+1) - p_t`.

It does not add trading signals and does not optimize for profit.

## 1. Does F_liq improve over p_t baseline consistently?

{consistent_text}

- liq_p_vs_p positive rate: {positive_rate:.2f}
- liq_p_vs_p mean delta_test_R2: {mean_delta:.6g}

## 2. Which timeframe works best?

Best interval for liq_p_vs_p: {_best_group(summary_interval, "interval", "liq_p_vs_p")}

## 3. Which profile method works best?

Best method for liq_p_vs_p: {_best_group(summary_method, "profile_method", "liq_p_vs_p")}

## 4. Barrier or well?

Best mode for liq_p_vs_p: {_best_group(summary_mode, "mode", "liq_p_vs_p")}

Barrier treats high historical volume as a resistance/barrier. Well treats it
as an attractor or fair-value zone.

## 5. Inside vs outside profile range

{outside_text}

Inside/outside deltas are computed from `metrics_inside_profile.csv` and
`metrics_outside_profile.csv` when sufficient samples exist.

## 6. Is there enough evidence to move to order-book v0.3?

{order_book_text}

The correct next scientific test is to replace OHLCV-derived volume profiles
with order-book depth or order-flow imbalance and repeat the same p_t-controlled
incremental analysis.

## Top Configurations

```text
{top_configs.head(20).to_string(index=False)}
```
"""
    (output_dir / "sweep_report.md").write_text(report, encoding="utf-8")


def _iter_configs(args: argparse.Namespace) -> Iterable[dict[str, object]]:
    for symbol in args.symbols:
        for interval in args.intervals:
            for profile_method in args.methods:
                for mode in args.modes:
                    for profile_window in args.profile_windows:
                        for bins in args.bins:
                            yield {
                                "symbol": symbol.upper(),
                                "interval": interval,
                                "profile_method": profile_method,
                                "mode": mode,
                                "profile_window": profile_window,
                                "bins": bins,
                            }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    raw_dir = PROJECT_ROOT / "data" / "raw"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for config in _iter_configs(args):
        symbol = str(config["symbol"])
        interval = str(config["interval"])
        profile_method = str(config["profile_method"])
        mode = str(config["mode"])
        profile_window = int(config["profile_window"])
        bins = int(config["bins"])

        data_path = raw_dir / f"{symbol}_{interval}.csv"
        run_name = _safe_name(symbol, interval, profile_method, mode, f"w{profile_window}", f"b{bins}")
        run_dir = runs_dir / run_name

        print(f"Running {run_name}")
        if not args.skip_existing or not data_path.exists():
            download_binance_ohlcv(
                symbol=symbol,
                interval=interval,
                output_path=data_path,
                limit=args.limit,
            )

        if not args.skip_existing or not (run_dir / "incremental_value.csv").exists():
            run_experiment(
                csv_path=str(data_path),
                output_dir=str(run_dir),
                volume_window=profile_window,
                profile_window=profile_window,
                bins=bins,
                potential_mode=mode,
                profile_method=profile_method,
                train_ratio=0.7,
                compare_modes=False,
                min_samples=100,
            )

        incremental = _read_csv_if_exists(run_dir / "incremental_value.csv")
        metrics_all = _read_csv_if_exists(run_dir / "metrics_all.csv")
        metrics_inside = _read_csv_if_exists(run_dir / "metrics_inside_profile.csv")
        metrics_outside = _read_csv_if_exists(run_dir / "metrics_outside_profile.csv")

        if incremental is None:
            print(f"Skipping missing incremental results for {run_name}")
            continue

        for comparison in TARGET_COMPARISONS:
            inc_rows = incremental.loc[incremental["comparison"] == comparison]
            if inc_rows.empty:
                continue
            inc = inc_rows.iloc[0].to_dict()
            inside = _comparison_delta(metrics_inside, comparison)
            outside = _comparison_delta(metrics_outside, comparison)
            all_delta = _comparison_delta(metrics_all, comparison)

            row = {
                "symbol": symbol,
                "interval": interval,
                "profile_method": profile_method,
                "mode": mode,
                "profile_window": profile_window,
                "bins": bins,
                "comparison": comparison,
                "delta_test_R2": inc.get("delta_test_R2"),
                "delta_RMSE": inc.get("delta_RMSE"),
                "delta_MAE": inc.get("delta_MAE"),
                "delta_sign_accuracy": inc.get("delta_sign_accuracy"),
                "conclusion": inc.get("conclusion"),
                "all_sample_size": all_delta.get("sample_size"),
                "inside_delta_test_R2": inside.get("delta_test_R2"),
                "inside_delta_RMSE": inside.get("delta_RMSE"),
                "inside_sample_size": inside.get("sample_size"),
                "inside_status": inside.get("status"),
                "outside_delta_test_R2": outside.get("delta_test_R2"),
                "outside_delta_RMSE": outside.get("delta_RMSE"),
                "outside_sample_size": outside.get("sample_size"),
                "outside_status": outside.get("status"),
                "run_dir": str(run_dir),
                "data_path": str(data_path),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("Sweep produced no result rows")

    results = pd.DataFrame(rows)
    results_path = output_dir / "sweep_results.csv"
    results.to_csv(results_path, index=False)

    summary_interval = _summarize(results, "interval")
    summary_method = _summarize(results, "profile_method")
    summary_mode = _summarize(results, "mode")
    top_configs = results.sort_values(["delta_test_R2", "delta_RMSE"], ascending=[False, False]).head(50)

    summary_interval.to_csv(output_dir / "sweep_summary_by_interval.csv", index=False)
    summary_method.to_csv(output_dir / "sweep_summary_by_method.csv", index=False)
    summary_mode.to_csv(output_dir / "sweep_summary_by_mode.csv", index=False)
    top_configs.to_csv(output_dir / "top_configs.csv", index=False)
    _write_report(output_dir, results, summary_interval, summary_method, summary_mode, top_configs)

    print(f"Saved sweep results to {results_path}")
    print("Top configs:")
    print(
        top_configs[
            [
                "symbol",
                "interval",
                "profile_method",
                "mode",
                "profile_window",
                "bins",
                "comparison",
                "delta_test_R2",
                "delta_RMSE",
                "conclusion",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
