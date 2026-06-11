"""Convenience feature-engineering entry points."""

from __future__ import annotations

import pandas as pd

from lagrangian_market.liquidity import add_liquidity_force
from lagrangian_market.mechanics import add_mechanics_features


def build_feature_set(
    df: pd.DataFrame,
    volume_window: int = 500,
    profile_window: int = 500,
    bins: int = 50,
    potential_mode: str = "barrier",
) -> pd.DataFrame:
    """Add mechanics and liquidity-force features to an OHLCV DataFrame."""
    out = add_mechanics_features(df, volume_window=volume_window, use_log_price=True)
    out = add_liquidity_force(
        out,
        profile_window=profile_window,
        bins=bins,
        normalize=True,
        mode=potential_mode,
    )
    return out
