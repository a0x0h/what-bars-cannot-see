"""Smoke and sanity tests for the validation and robustness analyses (synthetic data).

Run: python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic import aggregate_1h, make, make_1m  # noqa: E402
from msr import analysis, robust, validation  # noqa: E402
from msr.features import build_dataset  # noqa: E402
from msr.single import single_predictor_oos  # noqa: E402


def _data():
    m1 = make_1m(days=75, seed=5)
    df = aggregate_1h(m1)
    df["trades"] = (df["volume"] * 3).round() + 1
    X, Y, sets, groups = build_dataset(df, (1, 2, 4), m1=m1)
    return df, m1, X


def test_validation_tables_run_and_are_bounded():
    df, m1, X = _data()
    cv = validation.classifier_variants(df, m1)
    assert cv["pearson"].between(-1, 1).all() and cv["sign_agreement"].between(0, 1).all()
    ic = validation.information_ceiling(df, m1)
    # the nonparametric ceiling can never be far below a linear rescaling of BVC on the same data
    assert (ic["R2_ceiling_any_f(z)"] > ic["R2_BVC_rescaled"] - 0.05).all()
    ins = robust.ceiling_in_sample(df, m1)
    assert ins["R2_ceiling_in_sample"].between(0, 1).all()
    # in sample, the binned conditional mean weakly dominates any linear function of BVC
    assert (ins["R2_ceiling_in_sample"] >= ins["corr2_BVC"] - 0.02).all()
    sp = validation.spread_proxy_diagnostics(df, X)
    assert len(sp) == 3


def test_flow_response_and_controls_run():
    df, m1, X = _data()
    fr = analysis.flow_response(df, X, horizons=(1, 2))
    assert set(fr["spec"]) == {"taker", "bvc", "joint"}
    rc = robust.response_controls(df, X, m1, horizons=(1, 2))
    assert np.isfinite(rc["t"]).all()


def test_single_predictor_is_out_of_sample():
    """A predictor that is pure noise must not show a significant out-of-sample gain."""
    df, m1, X = _data()
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.standard_normal(len(X)), index=X.index)
    t = single_predictor_oos(noise, df, horizons=(1,), min_train=24 * 20, step=24 * 7)
    assert t["R2oos_%"].iloc[0] < 0.5
    tr = robust.single_predictor_trading(df, X, h=2, min_train=24 * 20, step=24 * 7)
    assert np.isfinite(tr["breakeven_bps"]).all()


def test_positive_control_planted_flow_signal():
    """On the synthetic series with a planted flow-reversal, observed flow should predict and
    technical momentum should not (expanding OLS, purged, out of sample)."""
    raw = make(n=6000, seed=2)
    df = raw.set_index(pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True))
    X, Y, _, _ = build_dataset(df, (1,))
    tak = single_predictor_oos(X["taker_imb_24"], df, horizons=(1,), min_train=2000, step=500)
    mom = single_predictor_oos(X["mom_168"], df, horizons=(1,), min_train=2000, step=500)
    assert tak["R2oos_%"].iloc[0] > mom["R2oos_%"].iloc[0]
    assert tak["CW_p"].iloc[0] < 0.05
