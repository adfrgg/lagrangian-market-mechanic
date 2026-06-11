"""Rolling volume-profile liquidity potential and force estimates."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


PROFILE_METHODS = {"close", "typical", "hlc3", "range_uniform", "range_triangular"}


def _clean_arrays(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_hist, dtype=float)
    v = np.asarray(volume_hist, dtype=float)
    mask = np.isfinite(x) & np.isfinite(v) & (v >= 0)
    return x[mask], v[mask]


def _safe_log(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    return np.where(arr > 0, np.log(arr), np.nan)


def _histogram_from_points(x: np.ndarray, volume: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    x_clean, v_clean = _clean_arrays(x, volume)
    if len(x_clean) < 2 or np.nanstd(x_clean) == 0.0 or np.sum(v_clean) <= 0.0:
        center = float(x_clean[0]) if len(x_clean) else 0.0
        centers = np.linspace(center - 1e-6, center + 1e-6, bins)
        return np.zeros(bins, dtype=float), centers
    hist, edges = np.histogram(x_clean, bins=bins, weights=v_clean)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist.astype(float), centers


def build_volume_profile(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
    bins: int,
    method: str = "close",
    high_hist: Optional[np.ndarray] = None,
    low_hist: Optional[np.ndarray] = None,
    close_hist: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a historical volume profile and return ``(liquidity, centers)``.

    ``close`` assigns each bar's full volume to the provided log-price.
    ``typical``/``hlc3`` assigns volume to log((high + low + close) / 3).
    ``range_uniform`` distributes volume uniformly across bins overlapping the
    bar's log-low/log-high interval. ``range_triangular`` uses triangular weights
    centered near close.
    """
    if bins < 3:
        raise ValueError("bins must be at least 3")
    if method not in PROFILE_METHODS:
        raise ValueError(f"profile method must be one of {sorted(PROFILE_METHODS)}")

    x = np.asarray(x_hist, dtype=float)
    volume = np.asarray(volume_hist, dtype=float)

    if method == "close":
        return _histogram_from_points(x, volume, bins)

    high_log = _safe_log(high_hist)
    low_log = _safe_log(low_hist)
    close_log = _safe_log(close_hist)

    if method in {"typical", "hlc3"}:
        if high_hist is None or low_hist is None or close_hist is None:
            return _histogram_from_points(x, volume, bins)
        high = np.asarray(high_hist, dtype=float)
        low = np.asarray(low_hist, dtype=float)
        close = np.asarray(close_hist, dtype=float)
        typical_price = (high + low + close) / 3.0
        return _histogram_from_points(_safe_log(typical_price), volume, bins)

    if high_log is None or low_log is None:
        return _histogram_from_points(x, volume, bins)

    valid = (
        np.isfinite(high_log)
        & np.isfinite(low_log)
        & np.isfinite(volume)
        & (volume >= 0)
        & (high_log >= low_log)
    )
    if close_log is not None:
        valid = valid & np.isfinite(close_log)
    high_log = high_log[valid]
    low_log = low_log[valid]
    close_center = close_log[valid] if close_log is not None else x[valid]
    volume = volume[valid]

    if len(volume) < 1 or np.sum(volume) <= 0.0:
        return _histogram_from_points(x, volume_hist, bins)

    min_x = float(np.nanmin(low_log))
    max_x = float(np.nanmax(high_log))
    if not np.isfinite(min_x) or not np.isfinite(max_x) or min_x == max_x:
        return _histogram_from_points(close_center, volume, bins)

    edges = np.linspace(min_x, max_x, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(bins, dtype=float)

    for low_x, high_x, center_x, vol in zip(low_log, high_log, close_center, volume):
        if not np.isfinite(vol) or vol <= 0:
            continue
        if high_x == low_x:
            idx = int(np.clip(np.searchsorted(edges, center_x, side="right") - 1, 0, bins - 1))
            hist[idx] += vol
            continue

        overlap = (edges[:-1] <= high_x) & (edges[1:] >= low_x)
        idxs = np.flatnonzero(overlap)
        if len(idxs) == 0:
            idx = int(np.clip(np.searchsorted(edges, center_x, side="right") - 1, 0, bins - 1))
            hist[idx] += vol
            continue

        if method == "range_uniform":
            weights = np.ones(len(idxs), dtype=float)
        else:
            distances = np.abs(centers[idxs] - center_x)
            max_distance = float(np.max(distances))
            weights = np.ones(len(idxs), dtype=float) if max_distance == 0 else (max_distance - distances + 1e-12)
        weights = weights / np.sum(weights)
        hist[idxs] += vol * weights

    return hist, centers


def compute_volume_profile_potential(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
    profile_method: str = "close",
    high_hist: Optional[np.ndarray] = None,
    low_hist: Optional[np.ndarray] = None,
    close_hist: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin centers, raw liquidity, and potential U(x,t)."""
    if mode not in {"barrier", "well"}:
        raise ValueError("mode must be 'barrier' or 'well'")

    liquidity, centers = build_volume_profile(
        x_hist=x_hist,
        volume_hist=volume_hist,
        bins=bins,
        method=profile_method,
        high_hist=high_hist,
        low_hist=low_hist,
        close_hist=close_hist,
    )

    if normalize:
        std = float(np.std(liquidity))
        potential = np.zeros_like(liquidity, dtype=float) if std == 0.0 else (liquidity - float(np.mean(liquidity))) / std
    else:
        potential = liquidity.astype(float)

    if mode == "well":
        potential = -potential

    return centers.astype(float), liquidity.astype(float), potential.astype(float)


def compute_liquidity_force_for_point(
    x_hist: np.ndarray,
    volume_hist: np.ndarray,
    x_now: float,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
    profile_method: str = "close",
    high_hist: Optional[np.ndarray] = None,
    low_hist: Optional[np.ndarray] = None,
    close_hist: Optional[np.ndarray] = None,
) -> float:
    """Estimate liquidity force at the current log-price."""
    if not np.isfinite(x_now):
        return np.nan

    centers, _liquidity, potential = compute_volume_profile_potential(
        x_hist=x_hist,
        volume_hist=volume_hist,
        bins=bins,
        normalize=normalize,
        mode=mode,
        profile_method=profile_method,
        high_hist=high_hist,
        low_hist=low_hist,
        close_hist=close_hist,
    )
    if len(centers) < 3 or np.allclose(potential, potential[0]):
        return 0.0

    gradient = np.gradient(potential, centers)
    idx = int(np.argmin(np.abs(centers - float(x_now))))
    force = -gradient[idx]
    return float(force) if np.isfinite(force) else np.nan


def _profile_entropy(liquidity: np.ndarray) -> float:
    total = float(np.sum(liquidity))
    if total <= 0.0:
        return 0.0
    probs = liquidity / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _profile_range(
    x_hist: np.ndarray,
    method: str,
    high_hist: Optional[np.ndarray],
    low_hist: Optional[np.ndarray],
    close_hist: Optional[np.ndarray],
) -> tuple[float, float]:
    if method in {"range_uniform", "range_triangular"} and high_hist is not None and low_hist is not None:
        high_log = _safe_log(high_hist)
        low_log = _safe_log(low_hist)
        values = np.concatenate([low_log[np.isfinite(low_log)], high_log[np.isfinite(high_log)]])
    elif method in {"typical", "hlc3"} and high_hist is not None and low_hist is not None and close_hist is not None:
        values = _safe_log((np.asarray(high_hist, dtype=float) + np.asarray(low_hist, dtype=float) + np.asarray(close_hist, dtype=float)) / 3.0)
        values = values[np.isfinite(values)]
    else:
        values = np.asarray(x_hist, dtype=float)
        values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.min(values)), float(np.max(values))


def add_liquidity_force(
    df: pd.DataFrame,
    profile_window: int = 500,
    bins: int = 50,
    normalize: bool = True,
    mode: str = "barrier",
    profile_method: str = "close",
) -> pd.DataFrame:
    """Add rolling liquidity force and profile diagnostics without lookahead."""
    if profile_window <= 0:
        raise ValueError("profile_window must be positive")
    if "volume" not in df.columns:
        raise ValueError("df must contain a volume column")
    if profile_method not in PROFILE_METHODS:
        raise ValueError(f"profile_method must be one of {sorted(PROFILE_METHODS)}")

    out = df.copy()
    if "x" not in out.columns:
        if "close" not in out.columns:
            raise ValueError("df must contain either x or close")
        out["x"] = np.log(out["close"].astype(float))

    x_values = out["x"].to_numpy(dtype=float)
    volume_values = out["volume"].to_numpy(dtype=float)
    high_values = out["high"].to_numpy(dtype=float) if "high" in out.columns else None
    low_values = out["low"].to_numpy(dtype=float) if "low" in out.columns else None
    close_values = out["close"].to_numpy(dtype=float) if "close" in out.columns else None

    n = len(out)
    forces = np.full(n, np.nan, dtype=float)
    bin_indices = np.full(n, np.nan, dtype=float)
    u_at_price = np.full(n, np.nan, dtype=float)
    profile_x_min = np.full(n, np.nan, dtype=float)
    profile_x_max = np.full(n, np.nan, dtype=float)
    outside_profile_range = np.full(n, False, dtype=bool)
    distance_to_profile_range = np.full(n, np.nan, dtype=float)
    profile_bin_width = np.full(n, np.nan, dtype=float)
    profile_num_nonzero_bins = np.full(n, np.nan, dtype=float)
    profile_entropy = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - profile_window)
        end = i
        if end - start < 2:
            continue

        x_hist = x_values[start:end]
        volume_hist = volume_values[start:end]
        high_hist = high_values[start:end] if high_values is not None else None
        low_hist = low_values[start:end] if low_values is not None else None
        close_hist = close_values[start:end] if close_values is not None else None
        x_now = x_values[i]

        centers, liquidity, potential = compute_volume_profile_potential(
            x_hist=x_hist,
            volume_hist=volume_hist,
            bins=bins,
            normalize=normalize,
            mode=mode,
            profile_method=profile_method,
            high_hist=high_hist,
            low_hist=low_hist,
            close_hist=close_hist,
        )
        if len(centers) >= 2:
            profile_bin_width[i] = float(np.nanmedian(np.diff(centers)))
        profile_num_nonzero_bins[i] = int(np.sum(liquidity > 0))
        profile_entropy[i] = _profile_entropy(liquidity)

        min_x, max_x = _profile_range(x_hist, profile_method, high_hist, low_hist, close_hist)
        profile_x_min[i] = min_x
        profile_x_max[i] = max_x
        if np.isfinite(x_now) and np.isfinite(min_x) and np.isfinite(max_x):
            outside = bool(x_now < min_x or x_now > max_x)
            outside_profile_range[i] = outside
            distance_to_profile_range[i] = 0.0 if not outside else min(abs(x_now - min_x), abs(x_now - max_x))

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
            profile_method=profile_method,
            high_hist=high_hist,
            low_hist=low_hist,
            close_hist=close_hist,
        )

    out["F_liq"] = forces
    out["U_at_price"] = u_at_price
    out["liquidity_bin_index"] = bin_indices
    out["profile_x_min"] = profile_x_min
    out["profile_x_max"] = profile_x_max
    out["outside_profile_range"] = outside_profile_range
    out["distance_to_profile_range"] = distance_to_profile_range
    out["profile_bin_width"] = profile_bin_width
    out["profile_num_nonzero_bins"] = profile_num_nonzero_bins
    out["profile_entropy"] = profile_entropy
    return out
