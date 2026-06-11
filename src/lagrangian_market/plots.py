"""Matplotlib plotting utilities for research outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lagrangian_market.liquidity import compute_volume_profile_potential


def _time_axis(df: pd.DataFrame) -> pd.Series | pd.Index:
    return df["timestamp"] if "timestamp" in df.columns else df.index


def _prepare_output(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_kinetic_activity(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot close price and kinetic activity over time."""
    path = _prepare_output(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = _time_axis(df)
    axes[0].plot(t, df["close"], color="black", linewidth=1.0)
    axes[0].set_title("Close Price")
    axes[0].set_ylabel("Close")
    axes[1].plot(t, df["K"], color="tab:blue", linewidth=0.9)
    axes[1].set_title("Kinetic Activity")
    axes[1].set_ylabel("K")
    axes[1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_momentum(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot market momentum p over time."""
    path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(_time_axis(df), df["p"], color="tab:green", linewidth=0.9)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title("Market Momentum")
    ax.set_ylabel("p = m u")
    ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_force_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter current liquidity force against next observed force."""
    path = _prepare_output(output_path)
    clean = df.dropna(subset=["F_liq", "F_next"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(clean["F_liq"], clean["F_next"], s=10, alpha=0.35, color="tab:purple")
    if len(clean) > 2 and clean["F_liq"].std() > 0:
        coef = np.polyfit(clean["F_liq"], clean["F_next"], deg=1)
        xs = np.linspace(clean["F_liq"].min(), clean["F_liq"].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_title("Observed Future Force vs Liquidity Force")
    ax.set_xlabel("F_liq(t)")
    ax.set_ylabel("F_obs(t+1)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """Plot actual and predicted future observed force over the test period."""
    path = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    t = _time_axis(test_df)
    ax.plot(t, test_df["F_next"], label="actual F_next", color="black", linewidth=0.9)
    ax.plot(t, pred, label="predicted F_next", color="tab:red", linewidth=0.9, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Actual vs Predicted Force")
    ax.set_ylabel("Force")
    ax.set_xlabel("Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_volume_profile_snapshot(
    df: pd.DataFrame,
    idx: int,
    profile_window: int,
    bins: int,
    output_path: str | Path,
    mode: str = "barrier",
) -> None:
    """Plot a rolling liquidity potential snapshot with current price marked."""
    if idx <= 1 or idx >= len(df):
        raise ValueError("idx must be inside the DataFrame and have past history")
    path = _prepare_output(output_path)
    start = max(0, idx - profile_window)
    x_hist = df["x"].iloc[start:idx].to_numpy(dtype=float)
    volume_hist = df["volume"].iloc[start:idx].to_numpy(dtype=float)
    centers, _liquidity, potential = compute_volume_profile_potential(
        x_hist=x_hist,
        volume_hist=volume_hist,
        bins=bins,
        normalize=True,
        mode=mode,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, potential, color="tab:blue", linewidth=1.2)
    ax.axvline(df["x"].iloc[idx], color="tab:red", linewidth=1.2, label="x(t)")
    ax.set_title("Rolling Volume Profile Potential Snapshot")
    ax.set_xlabel("log-price x")
    ax.set_ylabel("U(x,t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residuals(residuals: np.ndarray, output_path: str | Path) -> None:
    """Plot prediction residual distribution."""
    path = _prepare_output(output_path)
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(clean, bins=50, color="tab:gray", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("actual - predicted")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
