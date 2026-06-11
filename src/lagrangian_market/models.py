"""Regression models and chronological evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PredictiveModel(Protocol):
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return predictions for a model-specific feature frame."""


@dataclass
class ZeroModel:
    """Baseline model that predicts zero observed force."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x), dtype=float)


@dataclass
class ColumnPersistenceModel:
    """Baseline model that predicts the target from a current-period column."""

    column: str

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x[self.column].to_numpy(dtype=float)


@dataclass
class NegativePFixedModel:
    """Artifact baseline implied by F_next_t = p_{t+1} - p_t.

    If next-period momentum is approximately noise around zero, a mechanical
    predictor for the target is simply ``-p_t``.
    """

    column: str = "p"

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return -x[self.column].to_numpy(dtype=float)


@dataclass
class MajoritySignModel:
    """Baseline that always predicts the most common training target sign."""

    sign: float
    magnitude: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.sign * self.magnitude, dtype=float)


def train_test_split_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split a DataFrame into train and test partitions."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_ratio creates an empty train or test split")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _clean_supervised_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    cols = list(dict.fromkeys(feature_cols + [target_col]))
    return df.dropna(subset=cols).copy()


def fit_linear_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "F_next",
) -> LinearRegression:
    """Fit a scikit-learn linear regression on chronological training data."""
    clean = _clean_supervised_frame(train_df, feature_cols, target_col)
    if clean.empty:
        raise ValueError("No valid training rows after dropping NaNs")

    model = LinearRegression()
    model.fit(clean[feature_cols], clean[target_col])
    model.feature_cols_ = feature_cols  # type: ignore[attr-defined]
    model.target_col_ = target_col  # type: ignore[attr-defined]
    return model


def predict_model(
    model: PredictiveModel,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Predict with either a linear model or a simple baseline model."""
    if isinstance(model, (ZeroModel, ColumnPersistenceModel, NegativePFixedModel, MajoritySignModel)):
        return model.predict(df)
    return model.predict(df[feature_cols])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute out-of-sample regression and directional metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    pred = y_pred[mask]
    if len(y) == 0:
        raise ValueError("No finite rows available for evaluation")

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(pred) > 0 and np.std(y) > 0 else np.nan
    pred_sign = np.sign(pred)
    y_sign = np.sign(y)
    nonzero_pred_mask = pred_sign != 0
    sign_accuracy_nonzero = (
        float(np.mean(y_sign[nonzero_pred_mask] == pred_sign[nonzero_pred_mask]))
        if np.any(nonzero_pred_mask)
        else np.nan
    )

    return {
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse,
        "sign_accuracy": float(np.mean(y_sign == pred_sign)),
        "sign_accuracy_raw": float(np.mean(y_sign == pred_sign)),
        "sign_accuracy_nonzero_pred": sign_accuracy_nonzero,
        "corr": corr,
        "dtw_raw_norm": _normalized_dtw_distance(y, pred),
        "dtw_z_norm": _normalized_dtw_distance(_zscore(y), _zscore(pred)),
        "sample_size": float(len(y)),
    }


def evaluate_model(
    model: PredictiveModel,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "F_next",
) -> dict[str, float]:
    """Evaluate a model on chronological test data."""
    required_cols = list(dict.fromkeys(feature_cols + [target_col]))
    if isinstance(model, ColumnPersistenceModel):
        required_cols.append(model.column)
    if isinstance(model, NegativePFixedModel):
        required_cols.append(model.column)
    clean = test_df.dropna(subset=required_cols).copy()
    y_true = clean[target_col].to_numpy(dtype=float)
    y_pred = predict_model(model, clean, feature_cols)
    return evaluate_predictions(y_true, y_pred)


def coefficient_table(
    model: PredictiveModel,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return coefficients for a fitted linear model."""
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": np.asarray(model.coef_, dtype=float),
        }
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0:
        return np.full_like(values, np.nan, dtype=float)
    return (values - float(np.mean(values))) / std


def _normalized_dtw_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_ratio: float = 0.1,
) -> float:
    """Return DTW distance normalized by path length.

    DTW is useful here because force predictions may have the right shape with a
    small local timing offset. A Sakoe-Chiba band keeps the comparison local and
    avoids treating distant, unrelated events as matches.
    """
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    n = len(y)
    m = len(pred)
    if n == 0 or m == 0:
        return np.nan

    window = max(abs(n - m), int(max(n, m) * window_ratio), 1)
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf, dtype=float)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window) + 1
        for j in range(j_start, j_end):
            candidates = (
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            prev_cost, prev_steps = min(candidates, key=lambda item: item[0])
            cost[i, j] = abs(y[i - 1] - pred[j - 1]) + prev_cost
            steps[i, j] = prev_steps + 1

    if not np.isfinite(cost[n, m]) or steps[n, m] == 0:
        return np.nan
    return float(cost[n, m] / steps[n, m])
