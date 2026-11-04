"""Tests for the zero-fee natural-experiment code (synthetic data).

Run: python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msr import fee  # noqa: E402
from msr.events import ZERO_FEE_OFF, ZERO_FEE_ON, zero_fee_mask  # noqa: E402


def _minutes(days, start="2022-03-01", seed=0, churn=0.0, buy_share=None):
    """One-minute klines: random-walk prices, directional taker flow loaded on the return, and an
    optional block of perfectly two-sided volume `churn` (taker buys = taker sells)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    r = rng.normal(0, 5e-4, len(idx))
    c = 100 * np.exp(np.cumsum(r))
    net = 2e5 * r / 5e-4 + rng.normal(0, 1e5, len(idx))           # signed directional dollar flow
    gross_dir = np.abs(net) + 1e5
    buy = (gross_dir + net) / 2
    qv = gross_dir + churn
    tbq = buy + churn / 2
    if buy_share is not None:
        tbq = buy_share * qv
    d = pd.DataFrame({"high": c * 1.0002, "low": c * 0.9998, "close": c, "volume": qv / c, "quote_volume": qv,
                      "trades": np.maximum(1, qv / 1000).round(), "taker_buy_volume": tbq / c,
                      "taker_buy_quote_volume": tbq}, index=idx)
    return d


def test_ols_from_sums_matches_numpy():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=500), rng.normal(size=500)
    y = 0.7 * x + y
    s, r2 = fee._ols_from_sums(pd.Series([500.0]), pd.Series([x.sum()]), pd.Series([y.sum()]),
                               pd.Series([(x * x).sum()]), pd.Series([(y * y).sum()]), pd.Series([(x * y).sum()]))
    b = np.polyfit(x, y, 1)[0]
    assert abs(s.iloc[0] - b) < 1e-10
    assert abs(r2.iloc[0] - np.corrcoef(x, y)[0, 1] ** 2) < 1e-10


def test_two_sided_volume_is_separated_from_directional_volume():
    base = fee.daily_measures(_minutes(12, seed=2))
    churned = fee.daily_measures(_minutes(12, seed=2, churn=2e6))
    # perfectly two-sided volume raises dollar volume and two-sided volume but not directional volume
    assert np.allclose(base["net1m"], churned["net1m"], rtol=1e-9)
    assert (churned["dvol"] > 5 * base["dvol"]).all()
    assert (churned["ts_share"] > base["ts_share"]).all()
    # price impact per dollar of observed net flow is unchanged; volume-scaled proxies fall
    assert np.allclose(base["lam_true"], churned["lam_true"], rtol=1e-6)
    assert (churned["amihud"] < 0.3 * base["amihud"]).all()
    assert (churned["lam_bvc"] < 0.3 * base["lam_bvc"]).all()
    # all buys: directional volume equals dollar volume
    allbuy = fee.daily_measures(_minutes(3, seed=3, buy_share=1.0))
    assert np.allclose(allbuy["net1m"], allbuy["dvol"])
    assert np.allclose(allbuy["ts_share"], 0)


def test_did_recovers_planted_effect_and_placebos_do_not():
    rng = np.random.default_rng(4)
    days = pd.date_range("2021-10-01", "2026-10-31", freq="D", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.05, len(days)))                # market-wide shocks
    panel = {}
    for i in range(12):
        y = 10 + i * 0.1 + common + rng.normal(0, 0.1, len(days))
        if i == 0:
            y = y + 1.0 * fee._in(days, fee.ZERO)                     # treated: +1 log point while fee-free
        panel[f"C{i}"] = pd.DataFrame({"dvol": np.exp(y)}, index=days)
    tab = fee.did_table(panel, "C0", [f"C{i}" for i in range(1, 12)], keys=["dvol"])
    r = tab.iloc[0]
    assert abs(r["on"] - 1.0) < 0.05 and abs(r["off"] + 1.0) < 0.05
    assert r["on_t"] > 10 and r["p_space_on"] <= 1 / 12 + 1e-9 and r["p_time_on"] < 0.05
    # a control has no effect
    tab2 = fee.did_table(panel, "C3", [f"C{i}" for i in range(1, 12) if i != 3], keys=["dvol"])
    assert abs(tab2.iloc[0]["on"]) < 0.1


def test_event_times_and_mask():
    idx = pd.date_range("2022-07-08 12:00", periods=4, freq="h", tz="UTC")
    assert list(zero_fee_mask(idx)) == [False, False, True, True]
    assert ZERO_FEE_ON < ZERO_FEE_OFF
    assert not zero_fee_mask(pd.DatetimeIndex([ZERO_FEE_OFF]))[0]
