"""L2 model suite for Order-Book Lagrangian Market Mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_SPECS: dict[str, list[str]] = {
    "baseline_p_only": ["p"],
    "baseline_u_only": ["u"],
    "baseline_p_u": ["p", "u"],
    "baseline_flow_only": ["F_flow"],
    "baseline_book_imbalance": ["book_imbalance_20"],
    "l2_potential_only": ["F_pot_5bps"],
    "l2_potential_p": ["F_pot_5bps", "p"],
    "l2_potential_p_u": ["F_pot_5bps", "p", "u"],
    "l2_flow_p_u": ["F_flow", "p", "u"],
    "l2_full": ["F_pot_5bps", "F_flow", "book_imbalance_20", "p", "u", "spread_bps"],
    "l2_multi_horizon": ["F_pot_1bps", "F_pot_2bps", "F_pot_5bps", "F_pot_10bps", "F_flow", "p", "u"],
}


def train_test_split_time(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    split = int(len(df) * train_ratio)
    if split <= 0 or split >= len(df):
        raise ValueError("train_ratio creates an empty split")
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def baseline_negative_p_fixed(df: pd.DataFrame) -> np.ndarray:
    """Fixed artifact baseline y_hat = -p_t."""
    return -df["p"].to_numpy(dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    pred = y_pred[mask]
    if len(y) == 0:
        raise ValueError("No finite rows for evaluation")
    pred_sign = np.sign(pred)
    y_sign = np.sign(y)
    nonzero = pred_sign != 0
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0 else np.nan
    return {
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "sign_accuracy": float(np.mean(y_sign == pred_sign)),
        "sign_accuracy_nonzero_pred": float(np.mean(y_sign[nonzero] == pred_sign[nonzero])) if np.any(nonzero) else np.nan,
        "corr": corr,
        "sample_size": int(len(y)),
    }


def _fit_and_eval(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    clean_train = train_df.dropna(subset=features + [target_col])
    clean_test = test_df.dropna(subset=features + [target_col])
    model = LinearRegression()
    model.fit(clean_train[features], clean_train[target_col])
    train_pred = model.predict(clean_train[features])
    test_pred = model.predict(clean_test[features])
    train_metrics = _metrics(clean_train[target_col].to_numpy(dtype=float), train_pred)
    test_metrics = _metrics(clean_test[target_col].to_numpy(dtype=float), test_pred)
    row: dict[str, float | str] = {
        "model": name,
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "sign_accuracy": test_metrics["sign_accuracy"],
        "sign_accuracy_nonzero_pred": test_metrics["sign_accuracy_nonzero_pred"],
        "corr": test_metrics["corr"],
        "sample_size": test_metrics["sample_size"],
        "intercept": float(model.intercept_),
    }
    for feature, coefficient in zip(features, model.coef_):
        row[f"coef_{feature}"] = float(coefficient)
    coef_rows = [
        {"model": name, "feature": feature, "coefficient": float(coef), "intercept": float(model.intercept_)}
        for feature, coef in zip(features, model.coef_)
    ]
    return row, coef_rows


def run_l2_model_suite(df: pd.DataFrame, train_ratio: float = 0.7, target_col: str = "F_next") -> pd.DataFrame:
    """Run chronological L2 baselines and Lagrangian models."""
    required = sorted(set([target_col, "p"] + [feature for features in MODEL_SPECS.values() for feature in features]))
    clean = df.dropna(subset=required).copy()
    if len(clean) < 10:
        raise ValueError("Not enough rows after dropping NaNs")
    train_df, test_df = train_test_split_time(clean, train_ratio=train_ratio)

    rows: list[dict[str, float | str]] = []
    coef_rows: list[dict[str, float | str]] = []

    for name in ["baseline_zero", "baseline_negative_p_fixed"]:
        test_clean = test_df.dropna(subset=[target_col, "p"])
        train_clean = train_df.dropna(subset=[target_col, "p"])
        if name == "baseline_zero":
            train_pred = np.zeros(len(train_clean), dtype=float)
            test_pred = np.zeros(len(test_clean), dtype=float)
        else:
            train_pred = baseline_negative_p_fixed(train_clean)
            test_pred = baseline_negative_p_fixed(test_clean)
        train_metrics = _metrics(train_clean[target_col].to_numpy(dtype=float), train_pred)
        test_metrics = _metrics(test_clean[target_col].to_numpy(dtype=float), test_pred)
        rows.append(
            {
                "model": name,
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "sign_accuracy": test_metrics["sign_accuracy"],
                "sign_accuracy_nonzero_pred": test_metrics["sign_accuracy_nonzero_pred"],
                "corr": test_metrics["corr"],
                "sample_size": test_metrics["sample_size"],
                "intercept": 0.0 if name == "baseline_negative_p_fixed" else np.nan,
                "coef_p": -1.0 if name == "baseline_negative_p_fixed" else np.nan,
            }
        )
        if name == "baseline_negative_p_fixed":
            coef_rows.append({"model": name, "feature": "p", "coefficient": -1.0, "intercept": 0.0})

    for name, features in MODEL_SPECS.items():
        row, coefs = _fit_and_eval(name, train_df, test_df, features, target_col)
        rows.append(row)
        coef_rows.extend(coefs)

    results = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    results.attrs["coefficients"] = pd.DataFrame(coef_rows)
    return results


def l2_coefficient_table(results: pd.DataFrame) -> pd.DataFrame:
    coefficients = results.attrs.get("coefficients")
    if isinstance(coefficients, pd.DataFrame):
        return coefficients.copy()
    return pd.DataFrame(columns=["model", "feature", "coefficient", "intercept"])


def chronological_folds(n_rows: int, folds: int = 5, min_train_ratio: float = 0.4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding chronological folds."""
    if folds <= 0:
        return []
    min_train = max(2, int(n_rows * min_train_ratio))
    remaining = n_rows - min_train
    if remaining < folds:
        return []
    test_size = remaining // folds
    splits = []
    for fold in range(folds):
        train_end = min_train + fold * test_size
        test_end = n_rows if fold == folds - 1 else train_end + test_size
        if train_end < test_end:
            splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


def run_l2_fold_metrics(df: pd.DataFrame, folds: int = 5, target_col: str = "F_next") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-fold metrics and return fold rows plus fold summary."""
    required = sorted(set([target_col, "p"] + [feature for features in MODEL_SPECS.values() for feature in features]))
    clean = df.dropna(subset=required).copy().reset_index(drop=True)
    rows = []
    coefficient_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(chronological_folds(len(clean), folds=folds), start=1):
        train_df = clean.iloc[train_idx]
        test_df = clean.iloc[test_idx]
        fold_metrics = run_l2_model_suite(pd.concat([train_df, test_df]), train_ratio=len(train_df) / (len(train_df) + len(test_df)))
        fold_metrics["fold"] = fold_id
        rows.append(fold_metrics)
        coefs = l2_coefficient_table(fold_metrics)
        coefs["fold"] = fold_id
        coefficient_rows.append(coefs)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    fold_df = pd.concat(rows, ignore_index=True)
    summary = fold_df.groupby("model").agg(
        folds=("fold", "nunique"),
        mean_test_r2=("test_r2", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_sign_accuracy=("sign_accuracy", "mean"),
        positive_test_r2_rate=("test_r2", lambda s: float((s > 0).mean())),
    ).reset_index()
    if coefficient_rows:
        coefs_all = pd.concat(coefficient_rows, ignore_index=True)
        fpot = coefs_all.loc[coefs_all["feature"] == "F_pot_5bps"]
        if not fpot.empty:
            stability = fpot.groupby("model")["coefficient"].agg(
                fpot_coef_mean="mean",
                fpot_coef_sign_stability=lambda s: float(max((s > 0).mean(), (s < 0).mean())),
            ).reset_index()
            summary = summary.merge(stability, on="model", how="left")
    return fold_df, summary


def compute_l2_incremental_value(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare L2 models against matched p/u baselines."""
    comparisons = [
        ("l2_potential_p_vs_p", "l2_potential_p", "baseline_p_only"),
        ("l2_potential_p_u_vs_p_u", "l2_potential_p_u", "baseline_p_u"),
        ("l2_full_vs_p_u", "l2_full", "baseline_p_u"),
        ("l2_multi_horizon_vs_p_u", "l2_multi_horizon", "baseline_p_u"),
        ("l2_flow_p_u_vs_p_u", "l2_flow_p_u", "baseline_p_u"),
    ]
    by_model = metrics.set_index("model")
    rows = []
    for comparison, model_name, baseline_name in comparisons:
        if model_name not in by_model.index or baseline_name not in by_model.index:
            continue
        model = by_model.loc[model_name]
        baseline = by_model.loc[baseline_name]
        delta_r2 = float(model["test_r2"] - baseline["test_r2"])
        delta_rmse = float(baseline["rmse"] - model["rmse"])
        delta_mae = float(baseline["mae"] - model["mae"])
        delta_sign = float(model["sign_accuracy"] - baseline["sign_accuracy"])
        conclusion = "positive" if delta_r2 > 0 and delta_rmse > 0 else ("weak" if any(v > 0 for v in [delta_r2, delta_rmse, delta_mae, delta_sign]) else "negative")
        rows.append(
            {
                "comparison": comparison,
                "model": model_name,
                "baseline": baseline_name,
                "delta_test_R2": delta_r2,
                "delta_RMSE": delta_rmse,
                "delta_MAE": delta_mae,
                "delta_sign_accuracy": delta_sign,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)
