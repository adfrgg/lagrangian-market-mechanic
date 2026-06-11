"""Rolling volume-profile liquidity potential and force estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_arrays(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_hist, dtype=float)
    v = np.asarray(volume_hist, dtype=float)
    mask = np.isfinite(x) & np.isfinite(v) & (v >= 0)
    return x[mask], v[mask]


def compute_volume_profile_potential(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin centers, raw liquidity, and potential U(x,t).

    The raw liquidity landscape is a histogram of historical log-prices weighted
    by historical volume. In barrier mode, higher historical volume is treated as
    higher potential. In well mode, higher historical volume is treated as lower
    potential.
    """
    if bins < 3:
        raise ValueError("bins must be at least 3")
    if mode not in {"barrier", "well"}:
        raise ValueError("mode must be 'barrier' or 'well'")

    x, v = _clean_arrays(x_hist, volume_hist)
    if len(x) < 2 or np.nanstd(x) == 0.0 or np.sum(v) <= 0.0:
        centers = np.linspace(float(x[0]) - 1e-6, float(x[0]) + 1e-6, bins) if len(x) else np.arange(bins)
        return centers, np.zeros(bins, dtype=float), np.zeros(bins, dtype=float)

    liquidity, edges = np.histogram(x, bins=bins, weights=v)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if normalize:
        std = float(np.std(liquidity))
        if std == 0.0:
            potential = np.zeros_like(liquidity, dtype=float)
        else:
            potential = (liquidity - float(np.mean(liquidity))) / std
    else:
        potential = liquidity.astype(float)

    if mode == "well":
        potential = -potential

    return centers, liquidity.astype(float), potential.astype(float)


def compute_liquidity_force_for_point(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
    x_now: float,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
) -> float:
    """Estimate liquidity force at the current log-price.

    A rolling volume profile defines the potential ``U(x,t)``. The force is the
    negative potential gradient evaluated at the histogram bin nearest
    ``x_now``:

    ``F_liq = -dU/dx``.
    """
    if not np.isfinite(x_now):
        return np.nan

    centers, _liquidity, potential = compute_volume_profile_potential(
        x_hist=x_hist,
        volume_hist=volume_hist,
        bins=bins,
        normalize=normalize,
        mode=mode,
    )
    if len(centers) < 3 or np.allclose(potential, potential[0]):
        return 0.0

    gradient = np.gradient(potential, centers)
    idx = int(np.argmin(np.abs(centers - float(x_now))))
    force = -gradient[idx]
    return float(force) if np.isfinite(force) else np.nan


def add_liquidity_force(
    df: pd.DataFrame,
    profile_window: int = 500,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
) -> pd.DataFrame:
    """Add rolling liquidity force without lookahead.

    For row ``i``, the volume profile uses rows ``[i - profile_window, i)`` only.
    This strict-past convention supports using ``F_liq_i`` to explain or predict
    ``F_obs_{i+1}`` without future data leakage.
    """
    if profile_window <= 0:
        raise ValueError("profile_window must be positive")
    if "volume" not in df.columns:
        raise ValueError("df must contain a volume column")

    out = df.copy()
    if "x" not in out.columns:
        if "close" not in out.columns:
            raise ValueError("df must contain either x or close")
        out["x"] = np.log(out["close"].astype(float))

    x_values = out["x"].to_numpy(dtype=float)
    volume_values = out["volume"].to_numpy(dtype=float)
    forces = np.full(len(out), np.nan, dtype=float)
    bin_indices = np.full(len(out), np.nan, dtype=float)
    u_at_price = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        start = max(0, i - profile_window)
        end = i
        if end - start < 2:
            continue

        x_hist = x_values[start:end]
        volume_hist = volume_values[start:end]
        x_now = x_values[i]

        centers, _liquidity, potential = compute_volume_profile_potential(
            x_hist=x_hist,
            volume_hist=volume_hist,
            bins=bins,
            normalize=normalize,
            mode=mode,
        )
        if len(centers) >= 3 and np.isfinite(x_now):
            idx = int(np.argmin(np.abs(centers - x_now)))
            bin_indices[i] = idx
            u_at_price[i] = potential[idx]

        forces[i] = compute_liquidity_force_for_point(
            x_hist=x_hist,
            volume_hist=volume_hist,
            x_now=x_now,
            bins=bins,
            normalize=normalize,
            mode=mode,
        )

    out["F_liq"] = forces
    out["U_at_price"] = u_at_price
    out["liquidity_bin_index"] = bin_indices
    return out
