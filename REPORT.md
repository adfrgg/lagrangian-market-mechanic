# Lagrangian Market Mechanics v0.2 Report

## Purpose

This repository is a research prototype for testing whether a
physics-inspired liquidity potential can help explain changes in
volume-weighted price momentum. It is not a trading bot, does not emit buy/sell
signals, and does not optimize profit.

v0.2 addresses a critical audit issue in the original formulation:

```text
F_next_t = F_obs_{t+1} = p_{t+1} - p_t
```

Because the target contains `-p_t`, apparent damping can be an algebraic artifact
unless models are compared against momentum-aware baselines.

## Core Equations

```text
x_t = log(close_t)
u_t = x_t - x_{t-1}
m_t = volume_t / rolling_mean(volume_t)
p_t = m_t u_t
K_t = 0.5 m_t u_t^2
F_obs_t = p_t - p_{t-1}
F_next_t = p_{t+1} - p_t
```

The liquidity potential is estimated from a strict-past rolling volume profile:

```text
U(x,t) = rolling volume profile over log-price space
F_liq_t = -dU/dx
```

The strongest v0.2 test is:

```text
F_next_t = a + b F_liq_t + c p_t + d u_t + epsilon_t
```

`F_liq_t` is meaningful only if it improves out-of-sample performance after
controlling for `p_t`.

## v0.2 Additions

New artifact-control baselines:

- `baseline_negative_p_fixed`: predicts `-p_t` directly.
- `baseline_p_only`: learns `F_next_t = a + b p_t`.
- `baseline_u_only`: learns `F_next_t = a + b u_t`.
- `baseline_p_u`: controls for both current momentum and velocity.
- `majority_sign_baseline`: always predicts the most common training sign.

New Lagrangian variants:

- `lagrangian_liq_only`
- `lagrangian_liq_u`
- `lagrangian_liq_p`
- `lagrangian_liq_p_u`

New diagnostics:

- `sign_accuracy_raw`
- `sign_accuracy_nonzero_pred`
- `incremental_value.csv`
- inside-profile metrics
- outside-profile metrics
- profile entropy and nonzero-bin counts
- target-vs-negative-p plot
- F_liq-vs-p-residual plot

New profile construction methods:

- `close`
- `typical`
- `hlc3`
- `range_uniform`
- `range_triangular`

## No-Lookahead Convention

For row `i`, liquidity force uses only rows:

```text
[i - profile_window, i)
```

Future close, high, low, and volume do not affect `F_liq_i`. This is covered by
the pytest suite.

## Output Files

A normal v0.2 experiment writes:

```text
feature_data.csv
metrics_all.csv
metrics_inside_profile.csv
metrics_outside_profile.csv
incremental_value.csv
coefficients.csv
reports/report.md
figures/
```

The most important output is `incremental_value.csv`, which compares:

```text
lagrangian_liq_only  vs baseline_zero
lagrangian_liq_u     vs baseline_u_only
lagrangian_liq_p     vs baseline_p_only
lagrangian_liq_p_u   vs baseline_p_u
```

Positive `delta_test_R2` and positive `delta_RMSE` mean the liquidity model added
out-of-sample value relative to the matched baseline.

## Smoke-Run Result

Smoke command:

```powershell
python scripts/run_experiment.py --csv data/raw/btcusdt_1h.csv --output outputs/v02_smoke --volume-window 50 --profile-window 50 --bins 25 --profile-method range_uniform --mode barrier --min-samples 50 --train-ratio 0.7
```

Key results:

```text
rows after cleaning: 949
outside profile range: 6.22%
```

Top models:

```text
model                       test_r2   rmse      sign_accuracy   coef_F_liq   coef_p     coef_u
lagrangian_liq_p            0.488925  0.012495  0.740351        1e-06        -0.963751  NaN
lagrangian_liq_p_u          0.488532  0.012500  0.740351        1e-06        -0.950621 -0.046756
baseline_p_only             0.487137  0.012517  0.736842        NaN          -0.958950  NaN
baseline_p_u                0.487123  0.012517  0.736842        NaN          -0.958462 -0.001712
baseline_negative_p_fixed   0.485202  0.012540  0.719298        NaN          -1.000000  NaN
```

Incremental liquidity value:

```text
comparison       delta_test_R2  delta_RMSE  delta_MAE  delta_sign_accuracy  conclusion
liq_p_vs_p       0.001788       0.000022    0.000011   0.003509             positive
liq_p_u_vs_p_u   0.001409       0.000017    0.000025   0.003509             positive
liq_u_vs_u      -0.003464      -0.000035   -0.000049   0.024561             weak
liq_only_vs_zero -0.000583     -0.000005   -0.000042   0.498246             weak
```

## Interpretation

The v0.2 result changes the scientific interpretation:

- The dominant structure is current momentum `p_t`, not just velocity `u_t`.
- The fixed `-p_t` baseline is very strong, confirming the algebraic artifact
  concern was valid.
- In the smoke run, `F_liq` adds a small positive improvement after controlling
  for `p_t`.
- The improvement is real in the metric table, but small. It should not be
  overstated.

Correct framing:

> We are testing whether liquidity potential adds information beyond the
> mechanical reversal term implied by `F_next = p_{t+1} - p_t`.

Do not claim the model works unless `lagrangian_liq_p` or `lagrangian_liq_p_u`
consistently improves out-of-sample metrics beyond `baseline_p_only`,
`baseline_p_u`, and `baseline_negative_p_fixed`.

## How To Run

Install:

```powershell
pip install -e ".[dev]"
```

Download free Binance OHLCV:

```powershell
python scripts/download_binance_ohlcv.py --symbol BTCUSDT --interval 1h --limit 2000 --output data/raw/btcusdt_1h.csv
```

Run v0.2:

```powershell
python scripts/run_experiment.py --csv data/raw/btcusdt_1h.csv --output outputs/btc_v02 --volume-window 100 --profile-window 100 --bins 50 --profile-method range_uniform --mode barrier --train-ratio 0.7
```

Compare barrier and well:

```powershell
python scripts/run_experiment.py --csv data/raw/btcusdt_1h.csv --output outputs/btc_v02_compare --volume-window 100 --profile-window 100 --bins 50 --profile-method range_uniform --compare-modes
```

Run tests:

```powershell
python -m pytest
```

## Next Research Directions

The most important next step is replacing OHLCV volume profiles with better
liquidity proxies:

- L2 order book depth
- order-flow imbalance
- spread and market-impact proxies
- volatility/liquidity regimes
- walk-forward validation
- multiple assets and timeframes
- classification of force direction after p-control

OHLCV profile methods are useful for prototyping, but true liquidity potential
should come from order book data.
