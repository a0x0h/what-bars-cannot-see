"""Generate a synthetic hourly OHLCV file with a small, known predictable component.

Used only to test the pipeline end to end (tests and dry runs). The planted
signal: after hours of high signed-flow imbalance, the next hours partially
revert. A correct, leakage-free pipeline should find a small positive OOS R^2
for the order-flow features and roughly zero for pure momentum.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n=17520, seed=0, signal=0.12):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-07-25", periods=n, freq="1h", tz="UTC")
    vol = np.empty(n)
    vol[0] = 0.006
    eps = rng.standard_normal(n)
    flow = np.zeros(n)
    r = np.zeros(n)
    for i in range(1, n):
        vol[i] = np.sqrt(1e-7 + 0.08 * (r[i - 1] ** 2) + 0.9 * vol[i - 1] ** 2)
        flow[i] = 0.6 * flow[i - 1] + rng.standard_normal()
        past = flow[max(0, i - 4):i].mean()
        r[i] = vol[i] * (eps[i] - signal * past)
        r[i] += 0.002 * vol[i] * flow[i]  # contemporaneous flow-price link
    close = 60000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 1e-4, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 1, n)) * vol * 0.6)
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 1, n)) * vol * 0.6)
    volume = np.exp(6 + 0.5 * np.abs(flow) + rng.normal(0, 0.3, n) + 30 * vol)
    taker = volume * (0.5 + 0.1 * np.tanh(flow))
    return pd.DataFrame({"timestamp": ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")), "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": volume, "taker_buy_volume": taker})


def make_1m(days=60, seed=0):
    """1-minute bars (DatetimeIndex) with taker volume; for intrahour tests."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    r = rng.standard_normal(n) * 0.0008 + (rng.random(n) < 0.001) * rng.normal(0, 0.01, n)
    close = 90000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 3e-4, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 3e-4, n)))
    vol = np.exp(rng.normal(2, 0.5, n))
    taker = vol * np.clip(0.5 + 200 * r + rng.normal(0, 0.1, n), 0, 1)
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol,
                         "taker_buy_volume": taker}, index=t)


def aggregate_1h(m1: pd.DataFrame) -> pd.DataFrame:
    g = m1.groupby(m1.index.floor("h"))
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum(),
                         "taker_buy_volume": g["taker_buy_volume"].sum()})


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/SYNTH_1h.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print("wrote", out)
