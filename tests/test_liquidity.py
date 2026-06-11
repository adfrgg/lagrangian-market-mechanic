import numpy as np

from lagrangian_market.liquidity import compute_liquidity_force_for_point


def test_liquidity_force_is_finite() -> None:
    x_hist = np.linspace(1.0, 2.0, 100)
    volume_hist = np.linspace(10.0, 100.0, 100)

    force = compute_liquidity_force_for_point(x_hist, volume_hist, x_now=1.5, bins=10)

    assert np.isfinite(force)


def test_flat_liquidity_returns_zero() -> None:
    x_hist = np.ones(100)
    volume_hist = np.ones(100)

    force = compute_liquidity_force_for_point(x_hist, volume_hist, x_now=1.0, bins=10)

    assert abs(force) < 1e-12


def test_barrier_and_well_modes_have_opposite_sign() -> None:
    x_hist = np.linspace(0.0, 1.0, 200)
    volume_hist = 1.0 + 10.0 * x_hist
    x_now = 0.55

    barrier_force = compute_liquidity_force_for_point(
        x_hist,
        volume_hist,
        x_now=x_now,
        bins=20,
        mode="barrier",
    )
    well_force = compute_liquidity_force_for_point(
        x_hist,
        volume_hist,
        x_now=x_now,
        bins=20,
        mode="well",
    )

    assert np.isfinite(barrier_force)
    assert np.isfinite(well_force)
    assert barrier_force * well_force < 0
    assert np.isclose(barrier_force, -well_force)
