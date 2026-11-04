"""Economic evaluation of forecasts.

Overlapping-sleeve implementation (Jegadeesh & Titman 1993): a new signal is
formed every hour and held for h hours, with 1/h of capital per sleeve. The
position over (t, t+1] is the average of the last h signals, so every forecast
is used and no position overlaps are double counted. Transaction costs are
charged on every change of position: cost_t = c * |pos_t - pos_{t-1}|.

Two trading rules:
* "sign"      - always in the market, s_t = sign(f_t).
* "selective" - trade only when |f_t| exceeds the 80th percentile of the
                model's absolute in-sample forecasts (threshold re-estimated at
                each refit, never using OOS data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import block_bootstrap_idx

HOURS_PER_YEAR = 24 * 365


def signals(oos: pd.DataFrame, model: str, rule: str) -> pd.Series:
    f = oos[f"f_{model}"]
    if rule == "sign":
        return np.sign(f)
    thr = oos[f"thr_{model}"]
    return np.sign(f).where(f.abs() > thr, 0.0)


def positions(sig: pd.Series, h: int) -> pd.Series:
    return sig.rolling(h, min_periods=1).sum() / h


def pnl(oos: pd.DataFrame, pos: pd.Series, cost_bps: float) -> pd.Series:
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * oos["r1_next"] - turnover * cost_bps / 1e4


def perf_stats(ret: pd.Series, pos: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    ann_ret = eq.iloc[-1] ** (HOURS_PER_YEAR / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(HOURS_PER_YEAR)
    sharpe = ret.mean() / ret.std() * np.sqrt(HOURS_PER_YEAR) if ret.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    d = {"total_ret_%": 100 * (eq.iloc[-1] - 1), "ann_ret_%": 100 * ann_ret, "ann_vol_%": 100 * ann_vol,
         "sharpe": sharpe, "max_dd_%": 100 * mdd}
    if pos is not None:
        d["exposure_%"] = 100 * pos.abs().mean()
        d["turnover_per_day"] = pos.diff().abs().mean() * 24
    return d


def sharpe_ci(ret: pd.Series, block=168, n_boot=1000, seed=42):
    r = ret.dropna().values
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n_boot):
        x = r[block_bootstrap_idx(len(r), block, rng)]
        if x.std() > 0:
            s.append(x.mean() / x.std() * np.sqrt(HOURS_PER_YEAR))
    return tuple(np.percentile(s, [2.5, 97.5]))


def breakeven_cost_bps(oos, pos):
    gross = (pos * oos["r1_next"]).sum()
    turn = pos.diff().abs().fillna(pos.abs()).sum()
    return 1e4 * gross / turn if turn > 0 else np.nan


def backtest_table(oos: pd.DataFrame, models, h: int, costs_bps=(0, 2, 5, 10), rules=("selective", "sign")):
    rows, curves = [], {}
    for m in models:
        for rule in rules:
            pos = positions(signals(oos, m, rule), h)
            for c in costs_bps:
                ret = pnl(oos, pos, c)
                st = perf_stats(ret, pos)
                lo, hi = sharpe_ci(ret) if c == costs_bps[1] else (np.nan, np.nan)
                rows.append({"model": m, "rule": rule, "cost_bps": c, **st,
                             "sharpe_lo95": lo, "sharpe_hi95": hi,
                             "breakeven_bps": breakeven_cost_bps(oos, pos)})
                curves[(m, rule, c)] = (1 + ret.fillna(0)).cumprod()
    bh = oos["r1_next"].fillna(0)
    rows.append({"model": "buy_and_hold", "rule": "-", "cost_bps": 0, **perf_stats(bh)})
    curves[("buy_and_hold", "-", 0)] = (1 + bh).cumprod()
    return pd.DataFrame(rows), curves


def strategy_diagnostics(oos: pd.DataFrame, models, h: int, rules=("selective", "sign"), cost_bps: float = 0.0):
    """Exposure, market beta/alpha (HAC) and the deflated Sharpe ratio for every model x rule.

    The deflated Sharpe ratio (Bailey & Lopez de Prado 2014) is the probability
    that a strategy's true Sharpe exceeds the maximum expected among N = all
    model x rule trials under the null of zero skill, given their dispersion,
    the sample length and the return skewness/kurtosis.
    """
    import statsmodels.api as sm
    from scipy.stats import kurtosis, norm, skew
    rets, rows = {}, []
    for m in models:
        for rule in rules:
            sig = signals(oos, m, rule)
            pos = positions(sig, h)
            r = pnl(oos, pos, cost_bps).dropna()
            rets[(m, rule)] = r
            X = sm.add_constant(oos["r1_next"].reindex(r.index).fillna(0))
            res = sm.OLS(r.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
            yearly = r.groupby(r.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(HOURS_PER_YEAR))
            rows.append({"model": m, "rule": rule, "long_%": 100 * (sig > 0).mean(), "short_%": 100 * (sig < 0).mean(),
                         "flat_%": 100 * (sig == 0).mean(), "beta": res.params[1], "alpha_ann_%": 100 * res.params[0] * HOURS_PER_YEAR,
                         "alpha_t": res.tvalues[0], "sharpe": r.mean() / r.std() * np.sqrt(HOURS_PER_YEAR),
                         "sharpe_min_year": yearly.min(), "sharpe_max_year": yearly.max(),
                         "pos_years": f"{int((yearly > 0).sum())}/{len(yearly)}"})
    sr = np.array([r.mean() / r.std() for r in rets.values()])
    N, emc = len(sr), 0.5772156649
    sr0 = np.sqrt(sr.var(ddof=1)) * ((1 - emc) * norm.ppf(1 - 1 / N) + emc * norm.ppf(1 - 1 / (N * np.e)))
    for row, r in zip(rows, rets.values()):
        s, g3, g4 = r.mean() / r.std(), skew(r), kurtosis(r, fisher=False)
        den = np.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2)
        row["PSR_vs_0"] = float(norm.cdf(s * np.sqrt(len(r) - 1) / den))
        row["DSR"] = float(norm.cdf((s - sr0) * np.sqrt(len(r) - 1) / den))
        row["n_trials"] = N
        row["SR0_ann"] = sr0 * np.sqrt(HOURS_PER_YEAR)
    return pd.DataFrame(rows)
