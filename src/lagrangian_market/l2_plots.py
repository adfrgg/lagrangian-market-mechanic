"""Matplotlib plots for L2 Lagrangian Market Mechanics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _prepare(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _time(df: pd.DataFrame) -> pd.Series | pd.Index:
    return pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else df.index


def plot_price_mid_spread(df: pd.DataFrame, output_path: str | Path) -> None:
    path = _prepare(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(_time(df), df["mid"], color="black", linewidth=0.9)
    axes[0].set_title("Mid Price")
    axes[0].set_ylabel("mid")
    axes[1].plot(_time(df), df["spread_bps"], color="tab:red", linewidth=0.9)
    axes[1].set_title("Spread")
    axes[1].set_ylabel("bps")
    axes[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_potential_force_timeseries(df: pd.DataFrame, output_path: str | Path) -> None:
    path = _prepare(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(_time(df), df["F_pot_5bps"], color="tab:blue", linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_title("Order-Book Potential Force")
    axes[0].set_ylabel("F_pot_5bps")
    axes[1].plot(_time(df), df["F_flow"], color="tab:green", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_title("Aggressive Trade Flow Force")
    axes[1].set_ylabel("F_flow")
    axes[1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_observed_vs_predicted_force(test_df: pd.DataFrame, pred: np.ndarray, output_path: str | Path) -> None:
    path = _prepare(output_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(_time(test_df), test_df["F_next"], label="actual", color="black", linewidth=0.9)
    ax.plot(_time(test_df), pred, label="predicted", color="tab:red", linewidth=0.9, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Observed vs Predicted Force")
    ax.set_ylabel("F_next")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residual_added_value(df: pd.DataFrame, output_path: str | Path) -> None:
    path = _prepare(output_path)
    clean = df.dropna(subset=["F_next", "p", "u", "F_pot_5bps"])
    model = LinearRegression().fit(clean[["p", "u"]], clean["F_next"])
    residual = clean["F_next"] - model.predict(clean[["p", "u"]])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(clean["F_pot_5bps"], residual, s=10, alpha=0.35)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_title("F_pot vs Residual after p+u")
    ax.set_xlabel("F_pot_5bps")
    ax.set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fold_delta_r2(fold_metrics: pd.DataFrame, output_path: str | Path) -> None:
    path = _prepare(output_path)
    if fold_metrics.empty:
        return
    base = fold_metrics.loc[fold_metrics["model"] == "baseline_p_u", ["fold", "test_r2"]].rename(columns={"test_r2": "base_r2"})
    full = fold_metrics.loc[fold_metrics["model"] == "l2_full", ["fold", "test_r2"]].rename(columns={"test_r2": "full_r2"})
    merged = full.merge(base, on="fold")
    merged["delta_r2"] = merged["full_r2"] - merged["base_r2"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(merged["fold"].astype(str), merged["delta_r2"], color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Fold Delta R2: l2_full vs baseline_p_u")
    ax.set_xlabel("fold")
    ax.set_ylabel("delta R2")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_coefficient_stability(fold_metrics: pd.DataFrame, output_path: str | Path) -> None:
    path = _prepare(output_path)
    coef_col = "coef_F_pot_5bps"
    clean = fold_metrics.loc[(fold_metrics["model"] == "l2_full") & fold_metrics[coef_col].notna()]
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(clean["fold"], clean[coef_col], marker="o", color="tab:purple")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("F_pot Coefficient Stability")
    ax.set_xlabel("fold")
    ax.set_ylabel("coef F_pot_5bps")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_orderbook_potential_snapshot(df: pd.DataFrame, output_path: str | Path, idx: int | None = None) -> None:
    path = _prepare(output_path)
    if idx is None:
        idx = len(df) // 2
    row = df.iloc[idx]
    labels = ["down 10", "down 5", "down 2", "down 1", "up 1", "up 2", "up 5", "up 10"]
    values = [
        row.get("U_down_10bps", np.nan),
        row.get("U_down_5bps", np.nan),
        row.get("U_down_2bps", np.nan),
        row.get("U_down_1bps", np.nan),
        row.get("U_up_1bps", np.nan),
        row.get("U_up_2bps", np.nan),
        row.get("U_up_5bps", np.nan),
        row.get("U_up_10bps", np.nan),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, values, color=["tab:green"] * 4 + ["tab:red"] * 4)
    ax.set_title("Order-Book Potential Snapshot")
    ax.set_ylabel("quote notional")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
