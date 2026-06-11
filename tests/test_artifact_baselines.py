import numpy as np
import pandas as pd

from lagrangian_market.evaluation import run_model_suite
from lagrangian_market.models import NegativePFixedModel


def test_negative_p_fixed_predicts_negative_p() -> None:
    df = pd.DataFrame({"p": [0.1, -0.2, 0.0]})
    pred = NegativePFixedModel().predict(df)
    assert np.allclose(pred, [-0.1, 0.2, 0.0])


def test_model_suite_returns_artifact_control_models() -> None:
    n = 80
    p = np.linspace(-0.03, 0.03, n)
    u = p * 0.5 + 0.001
    f_liq = np.sin(np.linspace(0, 4, n))
    f_next = -0.8 * p + 0.02 * f_liq
    df = pd.DataFrame({"p": p, "u": u, "F_liq": f_liq, "F_next": f_next})

    results = run_model_suite(df, train_ratio=0.7)
    models = set(results["model"])

    assert "baseline_negative_p_fixed" in models
    assert "baseline_p_only" in models
    assert "baseline_u_only" in models
    assert "baseline_p_u" in models
    assert "lagrangian_liq_p" in models
    assert "lagrangian_liq_p_u" in models
    assert not results.loc[results["model"] == "baseline_p_only", "coef_p"].isna().all()
