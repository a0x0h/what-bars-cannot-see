"""Statistical inference for overlapping-horizon forecasts.

Overlapping h-hour targets make consecutive forecast errors serially
correlated, so every test uses Newey-West (HAC) standard errors or a moving
block bootstrap. Tests implemented:

* Out-of-sample R^2 against a benchmark (Campbell & Thompson 2008).
* Clark & West (2007) MSPE-adjusted test for nested models (one-sided).
* Diebold & Mariano (1995) equal-accuracy test (two-sided).
* Information coefficient (Spearman) with block-bootstrap confidence interval.
* Directional hit rate with HAC t-statistic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr


def nw_lags(h: int) -> int:
    return max(24, 2 * h)


def hac_mean(x, lags: int):
    """Mean of x with Newey-West t-stat. Returns (mean, t)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def r2_oos(y, f, b) -> float:
    y, f, b = map(np.asarray, (y, f, b))
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2))


def clark_west(y, f_bench, f_model, lags: int):
    """H0: model adds nothing over nested benchmark. Returns (t, one-sided p)."""
    y, fb, fm = map(np.asarray, (y, f_bench, f_model))
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    _, t = hac_mean(adj, lags)
    return t, float(1 - norm.cdf(t))


def diebold_mariano(y, f1, f2, lags: int):
    """d = e1^2 - e2^2; positive t means f2 is more accurate. Returns (t, two-sided p)."""
    y, f1, f2 = map(np.asarray, (y, f1, f2))
    d = (y - f1) ** 2 - (y - f2) ** 2
    _, t = hac_mean(d, lags)
    return t, float(2 * (1 - norm.cdf(abs(t))))


def block_bootstrap_idx(n: int, block: int, rng: np.random.Generator):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def ic_with_ci(y, f, block: int = 48, n_boot: int = 1000, seed: int = 42):
    y, f = np.asarray(y), np.asarray(f)
    ic = spearmanr(f, y).statistic
    if np.std(f) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = block_bootstrap_idx(len(y), block, rng)
        if np.std(f[i]) > 0:
            bs.append(spearmanr(f[i], y[i]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(ic), (float(lo), float(hi))


def hit_rate(y, f, lags: int):
    y, f = np.asarray(y), np.asarray(f)
    m = f != 0
    if m.sum() == 0:
        return float("nan"), float("nan")
    hits = (np.sign(f[m]) == np.sign(y[m])).astype(float)
    mean, t = hac_mean(hits - 0.5, lags)
    return mean + 0.5, t


def forecast_table(oos: pd.DataFrame, models: list[str], h: int) -> pd.DataFrame:
    lags = nw_lags(h)
    y = oos["y"].values
    rows = []
    for m in models:
        f = oos[f"f_{m}"].values
        row = {"model": m,
               "R2oos_vs_zero_%": 100 * r2_oos(y, f, oos["f_zero"]),
               "R2oos_vs_mean_%": 100 * r2_oos(y, f, oos["f_mean"])}
        if m not in ("zero",):
            t, p = clark_west(y, oos["f_zero"], f, lags) if m != "mean" else (np.nan, np.nan)
            row.update({"CW_t_vs_zero": t, "CW_p_vs_zero": p})
            ic, (lo, hi) = ic_with_ci(y, f) if np.std(f) > 0 else (np.nan, (np.nan, np.nan))
            hr, ht = hit_rate(y, f, lags) if m != "mean" else (np.nan, np.nan)
            row.update({"IC": ic, "IC_lo95": lo, "IC_hi95": hi, "hit_rate": hr, "hit_t": ht})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def cumulative_sse_diff(oos: pd.DataFrame, model: str, bench: str = "zero") -> pd.Series:
    """Goyal & Welch (2008) plot: cumulative SSE(benchmark) - SSE(model)."""
    e_b = (oos["y"] - oos[f"f_{bench}"]) ** 2
    e_m = (oos["y"] - oos[f"f_{model}"]) ** 2
    return (e_b - e_m).cumsum()


def predictive_regression(x: pd.Series, y: pd.Series, lags: int):
    """Univariate predictive regression y_{t,t+h} = a + b z(x_t) + e with HAC t."""
    d = pd.concat([x, y], axis=1).dropna()
    z = (d.iloc[:, 0] - d.iloc[:, 0].mean()) / d.iloc[:, 0].std()
    res = sm.OLS(d.iloc[:, 1].values, sm.add_constant(z.values)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[1]), float(res.tvalues[1])
