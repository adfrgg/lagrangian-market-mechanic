"""Physics-inspired market mechanics features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_mechanics_features(
    df: pd.DataFrame,
    volume_window: int = 500,
    use_log_price: bool = True,
) -> pd.DataFrame:
    """Add Lagrangian-style mechanics features to OHLCV data.

    Physics interpretation:
    - ``x`` is the price coordinate, using log-price by default.
    - ``u`` is price velocity, the first difference of ``x``.
    - ``m`` is effective market mass, using volume normalized by rolling mean
      volume to avoid raw-volume scale explosion.
    - ``p`` is market momentum: ``p = m * u``.
    - ``K`` is market kinetic activity: ``K = 0.5 * m * u^2``.
    - ``F_obs`` is observed market force, the discrete change in momentum.

    ``F_next`` is the next-period observed force and is intended as the
    supervised target for testing whether current liquidity force explains
    future observed force.
    """
    if volume_window <= 0:
        raise ValueError("volume_window must be positive")
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("df must contain close and volume columns")

    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["x"] = np.log(close) if use_log_price else close
    out["u"] = out["x"].diff()

    rolling_volume = volume.rolling(volume_window, min_periods=volume_window).mean()
    out["m"] = volume / rolling_volume.replace(0.0, np.nan)
    out["p"] = out["m"] * out["u"]
    out["K"] = 0.5 * out["m"] * out["u"] ** 2
    out["F_obs"] = out["p"].diff()
    out["F_next"] = out["F_obs"].shift(-1)

    return out
