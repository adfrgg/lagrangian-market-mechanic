# Lagrangian Market Mechanics v0.1

This project is a Python research framework for testing a physics-inspired
market-mechanics model on OHLCV data. It is not a trading bot, does not generate
buy/sell signals, and does not claim that markets obey physical laws.

The core idea is to treat price as a generalized coordinate moving through a
liquidity potential. Volume acts as an effective mass. The experiment tests
whether changes in volume-weighted price momentum relate to the gradient of a
rolling liquidity potential estimated from a volume profile.

## Core Equations

Use log-price as the coordinate:

```text
x_t = log(S_t)
u_t = x_t - x_{t-1}
m_t = V_t / rolling_mean(V_t)
p_t = m_t u_t
K_t = 0.5 m_t u_t^2
F_obs_t = p_t - p_{t-1}
```

The OHLCV-only liquidity potential is estimated from a rolling volume profile in
log-price space:

```text
U(x,t) = rolling volume profile over log-price bins
F_liq_t = -dU/dx
```

The tested regression equation is:

```text
F_obs_{t+1} = a + b F_liq_t - lambda u_t + epsilon_t
```

In code, the velocity coefficient is estimated directly as `coef_u`; negative
values are consistent with a damping/friction interpretation.

## Installation

```bash
cd lagrangian-market-mechanics
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Required CSV Format

The main input path is a CSV with these columns:

```text
timestamp, open, high, low, close, volume
```

The loader standardizes column names to lowercase, parses timestamps when
possible, sorts chronologically, removes rows with missing close or volume,
requires `close > 0`, and requires `volume >= 0`.

## Command-Line Example

First, download free Binance public candles:

```powershell
python scripts/download_binance_ohlcv.py ^
  --symbol BTCUSDT ^
  --interval 1h ^
  --limit 2000 ^
  --output data/raw/btcusdt_1h.csv
```

Then run the experiment:

```powershell
python scripts/run_experiment.py ^
  --csv data/raw/btcusdt_1h.csv ^
  --output outputs/run_001 ^
  --volume-window 100 ^
  --profile-window 100 ^
  --bins 50 ^
  --mode barrier ^
  --train-ratio 0.7
```

Use backslashes instead of carets on macOS/Linux.

For 2000 hourly candles, windows around 100 are a practical starting point. The
default 500-bar windows are better for larger files because rolling features
need enough past data before they become valid.

## Output Files

Each experiment writes:

- `metrics.csv`: baseline and Lagrangian model metrics.
- `feature_data.csv`: OHLCV data plus mechanics and liquidity features.
- `reports/report.md`: short research report and interpretation notes.
- `figures/kinetic_activity.png`: close price and kinetic activity.
- `figures/momentum.png`: market momentum.
- `figures/force_scatter.png`: `F_liq_t` vs `F_obs_{t+1}`.
- `figures/actual_vs_predicted.png`: test-period force prediction plot.
- `figures/volume_profile_snapshot.png`: one rolling potential snapshot.
- `figures/residuals.png`: residual histogram.

## Interpretation Guide

The main model is:

```text
F_obs_{t+1} = a + b F_liq_t + c u_t + epsilon_t
```

Interpret `b` as the empirical relationship between the volume-profile force and
future observed force. Interpret `c` as a velocity/friction proxy. Compare the
model against zero, momentum persistence, velocity-only, and liquidity-only
baselines. The useful evidence is out-of-sample, chronological test performance,
not in-sample fit.

Positive results would usually require:

- Test R2 above simple baselines.
- Sign accuracy above naive baselines.
- Stable coefficient signs across assets, windows, and regimes.
- Residuals without obvious structure.

Weak results are still informative. They may mean the OHLCV volume profile is too
coarse, gradients are too noisy, or the model needs order book data.

## Lookahead Convention

For row `i`, the liquidity profile uses only rows `[i - profile_window, i)`.
That is, `F_liq_i` uses strictly past observations and the current coordinate
`x_i`, then tests against `F_obs_{i+1}` through `F_next`.

## Known Limitations

- OHLCV volume profile is only a rough proxy for liquidity.
- True liquidity should come from order book data.
- Volume is not canonical mass.
- Markets are open, noisy, strategic, and non-conservative.
- The Lagrangian is an organizing framework, not a physical law.
- Derivatives are noisy at high frequency.
- Results require out-of-sample validation.

## Future Upgrades

- Use L2/L3 order book depth.
- Use order-flow imbalance as an external force.
- Add damping/friction terms from spread and microstructure cost.
- Test barrier versus well potential across assets.
- Run multi-asset and multi-timeframe experiments.
- Add volatility/liquidity regime analysis.
- Compare against OFI and MLOFI baselines.

## Development

Run tests with:

```bash
pytest
```

The package targets Python 3.10+ and uses pandas, numpy, scikit-learn,
matplotlib, pytest, and optional tqdm.
