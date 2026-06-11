# Lagrangian Market Mechanics v0.2

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
  --profile-method range_uniform ^
  --mode barrier ^
  --train-ratio 0.7
```

Use backslashes instead of carets on macOS/Linux.

For 2000 hourly candles, windows around 100 are a practical starting point. The
default 500-bar windows are better for larger files because rolling features
need enough past data before they become valid.

## Output Files

Each experiment writes:

- `metrics_all.csv`: full-sample baseline and Lagrangian model metrics.
- `metrics_inside_profile.csv`: metrics for samples inside the historical profile range.
- `metrics_outside_profile.csv`: metrics for outside-profile samples, or an insufficient-sample note.
- `incremental_value.csv`: matched liquidity-model improvements over artifact-control baselines.
- `coefficients.csv`: model coefficients and intercepts.
- `feature_data.csv`: OHLCV data plus mechanics and liquidity features.
- `reports/report.md`: short research report and interpretation notes.
- `figures/kinetic_activity.png`: close price and kinetic activity.
- `figures/momentum.png`: market momentum.
- `figures/force_scatter.png`: `F_liq_t` vs `F_obs_{t+1}`.
- `figures/actual_vs_predicted.png`: test-period force prediction plot.
- `figures/volume_profile_snapshot.png`: one rolling potential snapshot.
- `figures/residuals.png`: residual histogram.
- `figures/target_vs_negative_p.png`: mechanical `-p_t` baseline diagnostic.
- `figures/fliq_vs_p_residual.png`: liquidity force versus residual after p-control.
- `figures/force_scatter_inside_outside.png`: force scatter colored by profile-range status.
- `figures/profile_diagnostics.png`: outside-range, entropy, and nonzero-bin diagnostics.

## v0.2 Research Audit Fixes

The v0.1 target was:

```text
F_next_t = F_obs_{t+1} = p_{t+1} - p_t
```

That target mechanically contains `-p_t`. Therefore a model using `u_t` or any
variable correlated with current momentum can appear to discover damping even if
it mostly learns target arithmetic.

v0.2 adds the explicit mechanical baseline:

```text
F_next_t ~= -p_t
```

It also tests liquidity force after controlling for momentum:

```text
F_next_t = a + b F_liq_t + c p_t + epsilon_t
```

The key scientific question is now:

> Does `F_liq_t` add out-of-sample information beyond the mechanical reversal
> term implied by `F_next_t = p_{t+1} - p_t`?

The v0.2 model suite includes:

- `baseline_zero`
- `baseline_negative_p_fixed`
- `majority_sign_baseline`
- `baseline_p_only`
- `baseline_u_only`
- `baseline_p_u`
- `lagrangian_liq_only`
- `lagrangian_liq_u`
- `lagrangian_liq_p`
- `lagrangian_liq_p_u`

Interpretation rules:

- If `lagrangian_liq_only` works but `lagrangian_liq_p` does not, liquidity
  force may only be proxying for current momentum.
- If `lagrangian_liq_p` or `lagrangian_liq_p_u` improves test R2 and RMSE over
  the matched p-control baseline, that is stronger evidence that liquidity
  potential adds information.
- Do not claim the model works unless it improves beyond `baseline_negative_p_fixed`
  and `baseline_p_only` out of sample.

## Profile Methods

v0.2 supports multiple liquidity-profile construction methods:

- `close`: assigns each bar's volume to log(close).
- `typical` / `hlc3`: assigns volume to log((high + low + close) / 3).
- `range_uniform`: distributes volume uniformly across bins between log(low)
  and log(high).
- `range_triangular`: distributes volume across the bar range with higher weight
  near close.

`close` is the simplest and crudest method. `range_uniform` is a more realistic
OHLCV approximation because intrabar volume did not all occur at the close.

## Inside / Outside Profile Diagnostics

The rolling profile is built from historical prices. If current `x_t` lies
outside that historical range, the force is effectively evaluated near a
boundary bin. v0.2 adds:

```text
profile_x_min
profile_x_max
outside_profile_range
distance_to_profile_range
profile_bin_width
profile_num_nonzero_bins
profile_entropy
```

If a model works inside the profile but fails outside, the potential may be more
useful in liquidity-zone or mean-reverting regimes. If it works outside, it may
capture breakout behavior. If both fail, the OHLCV potential proxy is weak.

## Barrier vs Well

The potential mode controls how historical volume is interpreted:

- `barrier`: high historical volume is high potential, like resistance/barrier.
- `well`: high historical volume is low potential, like attractor/fair-value.

Run both modes with:

```powershell
python scripts/run_experiment.py --csv data/raw/btcusdt_1h.csv --output outputs/compare_modes --volume-window 100 --profile-window 100 --bins 50 --profile-method range_uniform --compare-modes
```

This writes:

```text
outputs/compare_modes/barrier/
outputs/compare_modes/well/
outputs/compare_modes/mode_comparison.csv
```

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

## Order-Book Lagrangian Market Mechanics v1.0

v0.1/v0.2 used rolling OHLCV volume profiles as a proxy for liquidity potential.
That was useful as a prototype and negative control, but it is not true
liquidity. Historical volume says where trades happened, not how hard price is
to move now.

v1.0 adds an order-book mechanics layer. The new hypothesis uses current book
depth as inertia and potential, and aggressive trade flow as an external force.

This is still not a trading strategy. There are no buy/sell signals, no profit
optimization, and no trading-return reports.

Physics-to-market dictionary:

| Physics | Market |
|---|---|
| position `x` | log mid-price |
| velocity `dx` | log mid-price change |
| mass / inertia `M` | local book depth or price-impact stiffness |
| momentum `p` | `M * dx` |
| kinetic activity `K` | `0.5 * M * dx^2` |
| potential `U` | cost to move through the order book |
| external force | aggressive trade flow / OFI proxy |
| friction | spread and impact |
| noise | random flow and news |

Core equations:

```text
best_bid_t = highest bid
best_ask_t = lowest ask
mid_t = (best_bid_t + best_ask_t) / 2
x_t = log(mid_t)
u_t = x_t - x_{t-1}
M_t = normalized local depth or impact stiffness
p_t = M_t u_t
K_t = 0.5 M_t u_t^2
F_obs_t = p_t - p_{t-1}
F_next_t = F_obs_{t+1}
```

Directional order-book potential:

```text
U_up_t(H) = cost to move upward by H bps through asks
U_down_t(H) = cost to move downward by H bps through bids
F_pot_t = log(U_down_t + eps) - log(U_up_t + eps)
```

Aggressive trade-flow force:

```text
F_flow_t = (taker_buy_quote - taker_sell_quote) / rolling_mean(total_trade_quote)
```

Empirical model:

```text
F_next_t =
  a
  + b1 F_pot_t
  + b2 F_flow_t
  + b3 book_imbalance_t
  + b4 p_t
  + b5 u_t
  + epsilon_t
```

Because `F_next_t = p_{t+1} - p_t` mechanically contains `-p_t`, every L2 model
is compared against `baseline_negative_p_fixed`, `baseline_p_only`, and
`baseline_p_u`.

### L2 Data Collection

Quick live collection, build, and experiment in one PowerShell line:

```powershell
python scripts/collect_binance_l2.py --symbols ETHUSDT --duration-minutes 15 --mode partial --depth-levels 20 --speed 100ms --output data/l2_raw/quick_eth; python scripts/build_l2_dataset.py --input data/l2_raw/quick_eth --symbol ETHUSDT --bar-size 5s --output data/l2_processed/ETHUSDT_5s_quick.parquet; python scripts/run_l2_experiment.py --data data/l2_processed/ETHUSDT_5s_quick.parquet --output outputs/l2_runs/ETHUSDT_5s_quick --folds 5
```

Fuller collection:

```powershell
python scripts/collect_binance_l2.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --duration-minutes 360 --mode partial --depth-levels 20 --speed 100ms --output data/l2_raw/l2_6h
```

Build a processed dataset:

```powershell
python scripts/build_l2_dataset.py --input data/l2_raw/l2_6h --symbol ETHUSDT --bar-size 5s --depth-levels 20 --horizon-bps 1 2 5 10 --mass-window 300 --output data/l2_processed/ETHUSDT_5s.parquet
```

Run one L2 experiment:

```powershell
python scripts/run_l2_experiment.py --data data/l2_processed/ETHUSDT_5s.parquet --output outputs/l2_runs/ETHUSDT_5s --train-ratio 0.7 --folds 5
```

Run validation across processed files:

```powershell
python scripts/run_l2_validation.py --processed-dir data/l2_processed --symbols BTCUSDT ETHUSDT SOLUSDT --bar-sizes 1s 5s 15s 60s --output outputs/l2_validation
```

First files to inspect:

- `outputs/l2_runs/<run>/incremental_value.csv`
- `outputs/l2_runs/<run>/metrics_all.csv`
- `outputs/l2_runs/<run>/fold_summary.csv`
- `outputs/l2_runs/<run>/report.md`
- `outputs/l2_runs/<run>/plots/residual_added_value.png`

The primary question is whether `F_pot` adds value beyond `p` and `u`. The
secondary question is whether `F_flow` adds value.

Known limitations:

- Partial depth streams are top-of-book snapshots, not full historical L3 data.
- Free Binance public data only exists from the moment collection starts.
- Parquet output falls back to CSV if no parquet engine is installed.
- Robust validation requires hours to days of data, multiple symbols, and
  chronological folds.
