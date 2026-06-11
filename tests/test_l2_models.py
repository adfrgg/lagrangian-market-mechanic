import numpy as np
import pandas as pd

from lagrangian_market.l2_models import baseline_negative_p_fixed, run_l2_model_suite, train_test_split_time


def _model_df(n: int = 120) -> pd.DataFrame:
    p = np.linspace(-0.02, 0.02, n)
    u = p * 0.4 + 0.001
    fpot = np.sin(np.linspace(0, 8, n)) * 0.1
    fflow = np.cos(np.linspace(0, 8, n)) * 0.1
    return pd.DataFrame(
        {
            "F_next": -0.9 * p + 0.03 * fpot + 0.02 * fflow,
            "p": p,
            "u": u,
            "F_flow": fflow,
            "book_imbalance_20": fpot * 0.5,
            "F_pot_1bps": fpot * 0.5,
            "F_pot_2bps": fpot * 0.8,
            "F_pot_5bps": fpot,
            "F_pot_10bps": fpot * 1.2,
            "spread_bps": np.full(n, 1.0),
        }
    )


def test_l2_model_suite_returns_required_models() -> None:
    results = run_l2_model_suite(_model_df(), train_ratio=0.7)
    models = set(results["model"])

    for model in [
        "baseline_zero",
        "baseline_negative_p_fixed",
        "baseline_p_only",
        "baseline_u_only",
        "baseline_p_u",
        "baseline_flow_only",
        "baseline_book_imbalance",
        "l2_potential_only",
        "l2_potential_p",
        "l2_potential_p_u",
        "l2_flow_p_u",
        "l2_full",
        "l2_multi_horizon",
    ]:
        assert model in models


def test_l2_negative_p_and_chronological_split() -> None:
    df = _model_df()
    pred = baseline_negative_p_fixed(df)
    assert np.allclose(pred, -df["p"])

    train, test = train_test_split_time(df, train_ratio=0.7)
    assert train.index.max() < test.index.min()
