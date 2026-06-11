# Lagrangian Market Mechanics v0.1: Project Report

## Executive Summary

This repository implements a Python research framework for testing a
physics-inspired market model on OHLCV data. The project does not implement a
trading strategy. It tests whether price, volume, momentum, kinetic activity,
and a rolling liquidity-potential estimate can be organized into an empirical
market-mechanics framework.

The main empirical result from the BTCUSDT 1h test is:

- A strong velocity damping / reversal effect appears consistently.
- The OHLCV-derived liquidity force adds only small incremental value.
- The combined model performs best, but most of the explanatory power comes
  from the velocity term, not the liquidity-potential term.

Best practical configuration from the tested grid:

```text
symbol: BTCUSDT
interval: 1h
window: 100
bins: 50
mode: barrier or well
model: lagrangian_liq_damping
test_r2: 0.309078
sign_accuracy: 0.755556
corr: 0.561984
coef_F_liq: +/- 6.511979e-07
coef_u: -2.026504
```

The fair conclusion is:

> The Lagrangian-style framework is useful as an organizing structure, but the
> current OHLCV rolling volume-profile proxy is not yet strong enough to prove a
> meaningful liquidity-potential effect beyond a simpler velocity damping
> baseline.

## Repository Structure

```text
lagrangian-market-mechanics/
  README.md
  REPORT.md
  requirements.txt
  pyproject.toml
  data/
    raw/
    processed/
  outputs/
    figures/
    reports/
    variation_grid_summary.csv
  models/
  notebooks/
    01_exploration.md
  scripts/
    download_binance_ohlcv.py
    run_experiment.py
    run_variation_grid.py
  src/
    lagrangian_market/
      __init__.py
      data.py
      features.py
      liquidity.py
      mechanics.py
      models.py
      evaluation.py
      plots.py
      experiment.py
  tests/
    test_mechanics.py
    test_liquidity.py
    test_no_lookahead.py
```

## Research Objective

The goal is to test a physics-inspired market model, not to predict profit or
generate trading signals.

The core research question:

> Does a liquidity-potential force estimated from rolling OHLCV volume profiles
> help explain future changes in volume-weighted price momentum?

In practical terms, the project asks whether:

```text
F_liq_t = -dU(x,t)/dx
```

contains useful information about:

```text
F_obs_{t+1}
```

where `F_obs` is the observed change in market momentum.

## Mathematical Model

The price coordinate is log-price:

```text
x_t = log(close_t)
```

Price velocity is the change in log-price:

```text
u_t = x_t - x_{t-1}
```

Volume is treated as effective mass, but normalized to avoid scale explosion:

```text
m_t = volume_t / rolling_mean(volume_t)
```

Market momentum:

```text
p_t = m_t u_t
```

Market kinetic activity:

```text
K_t = 0.5 m_t u_t^2
```

Observed market force:

```text
F_obs_t = p_t - p_{t-1}
```

The Lagrangian-style expression is:

```text
L(x, x_dot, t) = 0.5 m_t x_dot^2 - U(x,t)
```

The empirical equation tested is:

```text
F_obs_{t+1} = a + b F_liq_t + c u_t + epsilon_t
```

The `u_t` term is interpreted as a damping or friction proxy. A negative
coefficient on `u_t` means current price velocity tends to be followed by an
opposing force.

## Liquidity Potential Construction

Because this is v0.1 and uses only OHLCV data, true order book liquidity is not
available. The project approximates liquidity potential using a rolling volume
profile.

For each row `i`:

1. Look back over past rows only.
2. Build a histogram over historical log-prices.
3. Weight each log-price observation by volume.
4. Normalize the histogram.
5. Treat the normalized histogram as a potential field.
6. Compute the gradient over log-price bins.
7. Evaluate the force at the current log-price.

The strict no-lookahead convention is:

```text
F_liq_i uses rows [i - profile_window, i)
```

It does not use future price or future volume.

Two interpretations are implemented:

```text
barrier: U = normalized volume profile
well:    U = -normalized volume profile
```

For linear regression, barrier and well produce equivalent performance with
opposite coefficient signs.

## Data Source

The repository includes a free Binance downloader:

```text
scripts/download_binance_ohlcv.py
```

Example:

```powershell
python scripts/download_binance_ohlcv.py --symbol BTCUSDT --interval 1h --limit 2000 --output data/raw/btcusdt_1h.csv
```

The generated CSV schema is:

```text
timestamp, open, high, low, close, volume
```

The data loader:

- lowercases column names
- parses timestamps
- sorts chronologically
- removes missing close or volume
- requires `close > 0`
- requires `volume >= 0`

## Feature Pipeline

The main feature pipeline is:

```text
load_ohlcv()
add_mechanics_features()
add_liquidity_force()
run_model_suite()
```

Generated columns include:

```text
x
u
m
p
K
F_obs
F_next
F_liq
U_at_price
liquidity_bin_index
```

`F_next` is:

```text
F_next = F_obs.shift(-1)
```

This is the supervised learning target.

## Models Tested

The model suite includes:

```text
baseline_zero
baseline_momentum
baseline_velocity
lagrangian_liq_only
lagrangian_liq_damping
```

Model definitions:

```text
baseline_zero:
  F_next_hat = 0

baseline_momentum:
  F_next_hat = F_obs_t

baseline_velocity:
  F_next_hat = a + c u_t

lagrangian_liq_only:
  F_next_hat = a + b F_liq_t

lagrangian_liq_damping:
  F_next_hat = a + b F_liq_t + c u_t
```

All splits are chronological. No random train/test split is used.

## Evaluation Metrics

Primary metrics:

```text
test_r2
mae
rmse
sign_accuracy
corr
```

The main interpretation in this report uses `test_r2`, `sign_accuracy`, and
correlation.

DTW metrics were also added experimentally:

```text
dtw_raw_norm
dtw_z_norm
```

However, DTW is not treated as the primary measurement because it can reward
warped shape similarity even when direction and correlation are poor. In this
project, R2 and directional/correlation metrics are more interpretable for the
current pointwise force-regression setup.

## Main Single-Run Result

For BTCUSDT 1h with:

```text
volume_window = 100
profile_window = 100
bins = 50
mode = barrier
train_ratio = 0.7
```

Metrics:

```text
model                    train_r2   test_r2   sign_accuracy   corr      coef_F_liq    coef_u
lagrangian_liq_damping   0.312767   0.309078  0.755556        0.561984  6.512e-07    -2.0265
baseline_velocity        0.309988   0.308708  0.733333        0.561573  NaN          -2.0379
lagrangian_liq_only      0.007310   0.006442  0.544444        0.108275  1.054e-06     NaN
baseline_zero            ~0         ~0        0.000000        NaN       NaN           NaN
baseline_momentum       -1.996669  -1.805220  0.362963       -0.402699  NaN           NaN
```

Interpretation:

- The combined Lagrangian model is best by test R2 and sign accuracy.
- The velocity-only baseline is almost as good by test R2.
- Liquidity-only is weak.
- Momentum persistence performs very badly.
- The negative `coef_u` is strong and stable.

The strongest empirical effect is:

```text
coef_u ~= -2
```

This means the next observed force tends to oppose current price velocity.

## Variation Grid

The tested grid was:

```text
modes: barrier, well
windows: 50, 100, 200
bins: 25, 50, 100
```

The full summary is saved at:

```text
outputs/variation_grid_summary.csv
```

Top R2 results:

```text
mode     window  bins  model                    test_r2   sign_accuracy  coef_F_liq    coef_u
barrier  200     50    lagrangian_liq_damping   0.310233  0.733333       1.315e-06    -2.0920
well     200     50    lagrangian_liq_damping   0.310233  0.733333      -1.315e-06    -2.0920
barrier  200     25    lagrangian_liq_damping   0.310038  0.733333       1.521e-07    -2.0971
well     200     25    lagrangian_liq_damping   0.310038  0.733333      -1.521e-07    -2.0971
```

Best sign accuracy:

```text
mode     window  bins  model                    test_r2   sign_accuracy  coef_F_liq    coef_u
barrier  100     50    lagrangian_liq_damping   0.309078  0.755556       6.512e-07    -2.0265
well     100     50    lagrangian_liq_damping   0.309078  0.755556      -6.512e-07    -2.0265
```

Important comparison:

```text
window = 200, bins = 50:

lagrangian_liq_damping test_r2 = 0.310233
baseline_velocity      test_r2 = 0.309901
incremental gain       ~= 0.000333
```

The liquidity force improves the model only marginally over the velocity-only
baseline.

## Interpretation Of Results

The data supports this:

```text
Future observed force is strongly related to current velocity with opposite sign.
```

In market language, this resembles:

- short-horizon reversal
- damping
- friction
- mean reversion in force response

The data weakly supports this:

```text
OHLCV rolling volume-profile liquidity force contains some information.
```

But the support is not strong enough to claim that the liquidity-potential
gradient is a primary driver.

The current version does not strongly support this:

```text
F_liq alone explains future observed market force.
```

The liquidity-only model has low R2, weak correlation, and modest sign accuracy.

## Why The Current Liquidity Proxy Is Weak

The likely reason is that OHLCV volume profile is a rough proxy for liquidity.
It tells us where trading happened, not where resting liquidity currently exists.

Limitations:

- Historical volume is not the same as order book depth.
- Executed volume is not necessarily available liquidity.
- Volume profile bins are sensitive to window and bin size.
- Numerical gradients over histograms are noisy.
- The model does not include spread, order-flow imbalance, volatility regime, or
  market impact.
- BTCUSDT 1h candles may be too coarse for a force-style model.

## Code Quality And Safeguards

The project includes tests for:

- mechanics feature definitions
- liquidity force finite output
- barrier/well sign behavior
- no-lookahead behavior

The most important test is:

```text
test_no_lookahead.py
```

It changes future price and volume after a selected index, recomputes
`F_liq`, and verifies that the original `F_liq_i` does not change.

## How To Run

Install:

```powershell
cd C:\Intern\Nectec\Options_pricing\lagrangian-market-mechanics
pip install -e ".[dev]"
```

Download data:

```powershell
python scripts/download_binance_ohlcv.py --symbol BTCUSDT --interval 1h --limit 2000 --output data/raw/btcusdt_1h.csv
```

Run one experiment:

```powershell
python scripts/run_experiment.py --csv data/raw/btcusdt_1h.csv --output outputs/btcusdt_1h --volume-window 100 --profile-window 100 --bins 50 --mode barrier --train-ratio 0.7
```

Run the variation grid:

```powershell
python scripts/run_variation_grid.py --csv data/raw/btcusdt_1h.csv --output outputs/variation_grid_summary.csv --windows 50,100,200 --bins 25,50,100 --modes barrier,well --train-ratio 0.7
```

Run tests:

```powershell
python -m pytest
```

## Final Research Conclusion

This repository successfully builds and tests a v0.1 Lagrangian Market Mechanics
framework.

The framework is internally coherent:

- price is represented as log-coordinate
- volume is normalized as effective mass
- momentum and kinetic activity are measurable
- observed force is a momentum difference
- liquidity force is estimated from a rolling volume-profile potential
- no-lookahead constraints are enforced
- baseline comparisons are chronological

The empirical result is mixed:

- The damping term is consistently meaningful.
- The liquidity-potential force is weak in this OHLCV-only version.
- The combined model is best, but only slightly better than velocity alone.

The next research direction should probably not be more tuning of this exact
OHLCV histogram potential. A better approach would use richer microstructure
data or a more directly causal liquidity proxy.

## Suggested Next Approaches

Stronger future directions:

1. Use L2 order book depth instead of OHLCV volume profile.
2. Use order-flow imbalance as an external force.
3. Add spread and volatility as damping/friction terms.
4. Estimate potential from resting depth, not executed volume.
5. Test on shorter horizons, such as 1m or 5m crypto data.
6. Separate regimes by volatility and liquidity.
7. Compare against OFI, MLOFI, and simple autoregressive baselines.
8. Use walk-forward validation across multiple time periods.
9. Test multiple assets instead of one BTCUSDT sample.
10. Model force direction as classification instead of continuous regression.

The most promising next step is order book data. The current framework can be
kept, but `U(x,t)` should be replaced with a true liquidity/depth potential.
