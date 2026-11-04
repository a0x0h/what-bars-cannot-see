"""Leakage tests. Run: python -m pytest -q tests

1. Features at t must not change when every bar after t is replaced by noise.
2. Targets at t must use exactly close_{t+h}.
3. Every walk-forward training label must be observable before its test block.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic import make, make_1m  # noqa: E402
from msr.features import build_dataset, build_intrahour_features, build_ms_features, build_ti_features, \
    build_trade_features  # noqa: E402
from msr.walkforward import inner_split, make_folds  # noqa: E402


def _df(n=3000):
    raw = make(n=n, seed=1)
    df = raw.set_index(pd.to_datetime(raw.pop("timestamp"), unit="ms", utc=True))
    df["trades"] = (df["volume"] * 3).round()
    return df


def _perturb_after(df, t_pos, seed=7):
    rng = np.random.default_rng(seed)
    d = df.copy()
    k = len(d) - t_pos - 1
    scale = np.exp(rng.normal(0, 0.05, size=(k, 1)))
    d.iloc[t_pos + 1:, :4] = d.iloc[t_pos + 1:, :4].values * scale
    d.iloc[t_pos + 1:, 4:] = d.iloc[t_pos + 1:, 4:].values * rng.uniform(0.2, 5, size=(k, d.shape[1] - 4))
    return d


def test_features_are_causal():
    df = _df()
    for t_pos in [800, 1500, 2500]:
        d2 = _perturb_after(df, t_pos)
        for fn in (lambda x: build_ms_features(x)[0], build_ti_features, build_trade_features):
            a, b = fn(df).iloc[: t_pos + 1], fn(d2).iloc[: t_pos + 1]
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)


def test_intrahour_features_are_causal():
    m1 = make_1m(days=40, seed=3)
    hours = pd.date_range(m1.index[0].floor("h"), m1.index[-1].floor("h"), freq="1h", tz="UTC")
    cut = m1.index[30 * 1440 + 59]  # last minute of an hour
    m2 = m1.copy()
    after = m2.index > cut
    m2.loc[after, ["open", "high", "low", "close"]] *= 1.05
    m2.loc[after, "volume"] *= 3
    a = build_intrahour_features(m1, hours)
    b = build_intrahour_features(m2, hours)
    keep = hours <= cut.floor("h")
    pd.testing.assert_frame_equal(a[keep], b[keep], check_exact=False, rtol=1e-10, atol=1e-12)


def test_target_definition():
    df = _df(500)
    X, Y, _, _ = build_dataset(df, horizons=(4,))
    t = Y.index[10]
    expect = np.log(df["close"].loc[t + pd.Timedelta(hours=4)] / df["close"].loc[t])
    assert np.isclose(Y.loc[t, "y_4"], expect)


def test_walkforward_purge():
    idx = pd.date_range("2024-01-01", periods=20000, freq="1h", tz="UTC")
    for h in (1, 4, 24):
        for tr, te, s in make_folds(idx, h, 4380, 720):
            last_label_time = idx[tr].max() + pd.Timedelta(hours=h)
            assert last_label_time < idx[te].min(), "training label overlaps test block"
            inn, val = inner_split(idx[tr], h, 0.2)
            assert idx[tr][inn].max() + pd.Timedelta(hours=h) < idx[tr][val].min()
