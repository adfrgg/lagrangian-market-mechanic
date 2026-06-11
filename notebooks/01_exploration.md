# 01 Exploration: Lagrangian Market Mechanics

This notebook-style markdown walks through the first research workflow. Copy the
code blocks into Jupyter cells or run them in an interactive Python session.

## Model Setup

Price is modeled as a coordinate `x = log(close)`. Price velocity is `u = dx`.
Normalized volume is effective mass `m`. Market momentum is `p = m u`, kinetic
activity is `K = 0.5 m u^2`, and observed force is `F_obs = dp`.

The empirical test asks whether the liquidity potential force `F_liq = -dU/dx`
helps explain `F_obs(t+1)`.

```python
from pathlib import Path

import matplotlib.pyplot as plt

from lagrangian_market.data import load_ohlcv
from lagrangian_market.evaluation import run_model_suite
from lagrangian_market.liquidity import add_liquidity_force
from lagrangian_market.mechanics import add_mechanics_features
```

## Load OHLCV Data

```python
csv_path = Path("../data/raw/sample.csv")
df = load_ohlcv(csv_path)
df.head()
```

## Mechanics Features

```python
df = add_mechanics_features(df, volume_window=500)
df[["timestamp", "close", "x", "u", "m", "p", "K", "F_obs", "F_next"]].tail()
```

## Liquidity Force

The liquidity profile at row `i` uses rows `[i-profile_window, i)` only. This
avoids lookahead when testing against `F_obs(t+1)`.

```python
df = add_liquidity_force(
    df,
    profile_window=500,
    bins=50,
    normalize=True,
    mode="barrier",
)
feature_df = df.dropna(subset=["F_liq", "F_next", "u", "p", "K", "F_obs"]).copy()
feature_df[["timestamp", "F_liq", "F_obs", "F_next"]].tail()
```

## Visualize K, p, F_liq, and F_obs

```python
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
t = feature_df["timestamp"]

axes[0].plot(t, feature_df["K"], color="tab:blue")
axes[0].set_title("Kinetic Activity K")

axes[1].plot(t, feature_df["p"], color="tab:green")
axes[1].axhline(0, color="black", linewidth=0.7)
axes[1].set_title("Market Momentum p")

axes[2].plot(t, feature_df["F_liq"], color="tab:purple")
axes[2].axhline(0, color="black", linewidth=0.7)
axes[2].set_title("Liquidity Force F_liq")

axes[3].plot(t, feature_df["F_obs"], color="tab:red")
axes[3].axhline(0, color="black", linewidth=0.7)
axes[3].set_title("Observed Force F_obs")

fig.tight_layout()
plt.show()
```

## Run Model Suite

```python
metrics = run_model_suite(feature_df, train_ratio=0.7)
metrics
```

## Scatter: Liquidity Force vs Future Observed Force

```python
clean = feature_df.dropna(subset=["F_liq", "F_next"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(clean["F_liq"], clean["F_next"], s=10, alpha=0.35)
ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlabel("F_liq(t)")
ax.set_ylabel("F_obs(t+1)")
ax.set_title("Liquidity Force vs Future Observed Force")
plt.show()
```

## Interpretation

The first research question is not whether this makes money. The question is
whether a rolling liquidity potential organizes price, volume, momentum, kinetic
activity, and observed force into a measurable out-of-sample relationship. If
the liquidity-force models do not beat simple baselines, the current OHLCV
potential is not yet carrying enough useful information.
