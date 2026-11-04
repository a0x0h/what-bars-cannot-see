"""Referee-style robustness checks for the order-flow results.

* response_controls      Is the reversal after observed flow just a return reversal or a volume /
                         volatility effect? Adds lagged returns, volume and volatility controls, and
                         a horse race against the best available proxy (BVC on 1-minute bars).
* single_predictor_extra Out-of-sample test of single predictors including 1-minute BVC, with the
                         out-of-sample R^2 reported separately for each half of the test period.
* single_predictor_trading
                         Economic value of the one signal that is significant out of sample:
                         overlapping-sleeve rule on the observed-imbalance forecast, with costs and
                         the break-even cost.
* ceiling_sensitivity    Information ceiling with 20/50/100 bins and with the fitting and evaluation
                         halves swapped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from .events import zero_fee_mask
from .inference import clark_west, nw_lags, r2_oos
from .walkforward import make_folds

EPS = 1e-12


def bvc_1m_hourly(m1: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """BVC applied to 1-minute bars and aggregated to the hour (causal: minutes of hour t only)."""
    r1 = np.log(m1["close"]).diff()
    s1 = r1.rolling(1440, min_periods=360).std()
    sv = (2 * norm.cdf(r1 / (s1 + EPS)) - 1) * m1["volume"]
    hr = m1.index.floor("h")
    imb = sv.groupby(hr).sum() / (m1["volume"].groupby(hr).sum() + EPS)
    return imb.reindex(index).rename("bvc_1m_1")


def _std(s):
    return (s - s.mean()) / s.std()


def response_controls(df: pd.DataFrame, X: pd.DataFrame, m1: pd.DataFrame | None, horizons=(1, 2, 4)) -> pd.DataFrame:
    lc = np.log(df["close"])
    r = lc.diff()
    z = r / r.rolling(168, min_periods=48).std()
    base = pd.DataFrame({"taker": X["taker_imb_1"], "z": z.reindex(X.index),
                         "r_lag1": r.shift(1).reindex(X.index), "r_lag2": r.shift(2).reindex(X.index),
                         "r_lag3": r.shift(3).reindex(X.index), "mom_24": X["mom_24"],
                         "volume_z": X["volume_z_168"], "vol": X["gk_vol_24"], "taker_lag1": X["taker_imb_1"].shift(1)})
    if m1 is not None:
        base["bvc_1m"] = bvc_1m_hourly(m1, X.index)
    specs = {
        "(1) taker": ["taker"],
        "(2) + bar return": ["taker", "z"],
        "(3) + lagged returns, 24h momentum": ["taker", "z", "r_lag1", "r_lag2", "r_lag3", "mom_24"],
        "(4) + volume, volatility, lagged taker": ["taker", "z", "r_lag1", "r_lag2", "r_lag3", "mom_24",
                                                   "volume_z", "vol", "taker_lag1"],
    }
    if "bvc_1m" in base:
        specs["(5) 1m BVC alone"] = ["bvc_1m"]
        specs["(6) taker vs 1m BVC"] = ["taker", "bvc_1m"]
    rows = []
    for k in horizons:
        y = ((lc.shift(-k) - lc) * 1e4).reindex(X.index).rename("y")
        for name, cols in specs.items():
            d = pd.concat([y, base[cols]], axis=1).dropna()
            Z = d[cols].apply(_std)
            res = sm.OLS(d["y"].values, sm.add_constant(Z.values)).fit(cov_type="HAC", cov_kwds={"maxlags": max(24, 2 * k)})
            for j, c in enumerate(cols):
                if c in ("taker", "bvc_1m"):
                    rows.append({"k": k, "spec": name, "regressor": c, "beta_bps": res.params[j + 1],
                                 "t": res.tvalues[j + 1], "n": len(d), "r2_%": 100 * res.rsquared})
    return pd.DataFrame(rows)


def _single_oos(x: pd.Series, lc: pd.Series, h: int, min_train: int, step: int):
    y = (lc.shift(-h) - lc).reindex(x.index)
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    preds = []
    for tr, te, _ in make_folds(d.index, h, min_train, step):
        a = d[tr]
        mu, sd = a["x"].mean(), a["x"].std()
        zz = (a["x"] - mu) / sd
        b = float((zz * a["y"]).sum() / (zz * zz).sum())
        preds.append(pd.DataFrame({"f": b * (d.loc[te, "x"] - mu) / sd,
                                   "thr": np.quantile(np.abs(b * zz), 0.8)}, index=d.index[te]))
    f = pd.concat(preds)
    f["y"] = d.loc[f.index, "y"]
    return f


def single_predictor_extra(df, X, m1, horizons=(1, 2, 4), min_train=4380, step=720):
    lc = np.log(df["close"])
    preds = {"Observed taker imbalance": X["taker_imb_1"], "BVC imbalance (1h bars)": X["ofi_1"]}
    if m1 is not None:
        preds["BVC imbalance (1m bars)"] = bvc_1m_hourly(m1, X.index)
    rows = []
    for name, x in preds.items():
        for h in horizons:
            f = _single_oos(x, lc, h, min_train, step)
            t, p = clark_west(f["y"], np.zeros(len(f)), f["f"], nw_lags(h))
            half = len(f) // 2
            r = {"predictor": name, "horizon_h": h, "n_oos": len(f),
                 "R2oos_%": 100 * r2_oos(f["y"], f["f"], np.zeros(len(f))), "CW_t": t, "CW_p": p,
                 "IC": f["f"].corr(f["y"], method="spearman")}
            for lab, part in [("first_half", f.iloc[:half]), ("second_half", f.iloc[half:])]:
                r[f"R2oos_{lab}_%"] = 100 * r2_oos(part["y"], part["f"], np.zeros(len(part)))
                r[f"CW_p_{lab}"] = clark_west(part["y"], np.zeros(len(part)), part["f"], nw_lags(h))[1]
            rows.append(r)
    return pd.DataFrame(rows)


def single_predictor_trading(df, X, h=2, costs=(0, 1, 2, 5), min_train=4380, step=720):
    """Overlapping-sleeve rule on the out-of-sample observed-imbalance forecast."""
    lc = np.log(df["close"])
    f = _single_oos(X["taker_imb_1"], lc, h, min_train, step)
    r1 = (df["close"].shift(-1) / df["close"] - 1).reindex(f.index)
    rows = []
    for rule in ("sign", "selective"):
        s = np.sign(f["f"]) if rule == "sign" else np.sign(f["f"]).where(f["f"].abs() > f["thr"], 0.0)
        pos = s.rolling(h, min_periods=1).sum() / h
        turn = pos.diff().abs().fillna(pos.abs())
        gross = pos * r1
        be = 1e4 * gross.sum() / turn.sum()
        for c in costs:
            ret = (gross - turn * c / 1e4).dropna()
            rows.append({"rule": rule, "holding_h": h, "cost_bps": c,
                         "ann_ret_%": 100 * ret.mean() * 24 * 365,
                         "sharpe": ret.mean() / ret.std() * np.sqrt(24 * 365),
                         "turnover_per_day": turn.mean() * 24, "breakeven_bps": be,
                         "gross_bps_per_trade_unit": be})
    return pd.DataFrame(rows)


def ceiling_sensitivity(df: pd.DataFrame, m1: pd.DataFrame | None = None) -> pd.DataFrame:
    out = []

    def one(label, r, v, taker, w):
        sig = r.rolling(w, min_periods=max(20, w // 4)).std()
        d = pd.DataFrame({"z": r / (sig + EPS), "imb": (2 * taker - v) / (v + EPS),
                          "v": v}).dropna()
        d = d[d["v"] > 0]
        half = len(d) // 2
        for order, (tr, te) in {"fit 1st, eval 2nd": (d.iloc[:half], d.iloc[half:]),
                                "fit 2nd, eval 1st": (d.iloc[half:], d.iloc[:half])}.items():
            for nb in (20, 50, 100):
                edges = np.unique(np.quantile(tr["z"], np.linspace(0, 1, nb + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                cm = tr.groupby(pd.cut(tr["z"], edges), observed=False)["imb"].mean()
                fit = pd.cut(te["z"], edges).map(cm).astype(float)
                sst = ((te["imb"] - te["imb"].mean()) ** 2).sum()
                out.append({"bars": label, "split": order, "bins": nb,
                            "R2_ceiling": 1 - ((te["imb"] - fit) ** 2).sum() / sst})

    one("1h", np.log(df["close"]).diff(), df["volume"], df["taker_buy_volume"], 168)
    if m1 is not None:
        one("1m", np.log(m1["close"]).diff(), m1["volume"], m1["taker_buy_volume"], 1440)
    return pd.DataFrame(out)


def ceiling_in_sample(df: pd.DataFrame, m1: pd.DataFrame | None = None, n_bins: int = 50) -> pd.DataFrame:
    """Upper bound: in-sample R^2 of the binned conditional mean E[imb | z] within each period.

    With 50 bins and thousands of hours per period the overfitting bias is negligible, so this is
    an upper bound on what any return-based classifier calibrated to that period can explain.
    Also reports the slope of observed imbalance on BVC within the period (its calibration).
    """
    out = []

    def one(label, r, v, taker, w):
        sig = r.rolling(w, min_periods=max(20, w // 4)).std()
        z = r / (sig + EPS)
        d = pd.DataFrame({"z": z, "imb": (2 * taker - v) / (v + EPS), "bvc": 2 * norm.cdf(z) - 1, "v": v}).dropna()
        d = d[d["v"] > 0]
        t = d.index
        zf = zero_fee_mask(t)
        periods = {"full sample": np.ones(len(d), bool), "zero-fee window": zf, "standard fees": ~zf}
        for y in sorted(set(t.year)):
            periods[str(y)] = t.year == y
        for name, m in periods.items():
            g = d[m]
            if len(g) < 1000:
                continue
            edges = np.unique(np.quantile(g["z"], np.linspace(0, 1, n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            fit = g.groupby(pd.cut(g["z"], edges), observed=False)["imb"].transform("mean")
            sst = ((g["imb"] - g["imb"].mean()) ** 2).sum()
            out.append({"bars": label, "period": name, "n": len(g),
                        "R2_ceiling_in_sample": 1 - ((g["imb"] - fit) ** 2).sum() / sst,
                        "corr2_BVC": g["bvc"].corr(g["imb"]) ** 2,
                        "slope_imb_on_BVC": np.polyfit(g["bvc"], g["imb"], 1)[0],
                        "sd_imb": g["imb"].std()})

    one("1h", np.log(df["close"]).diff(), df["volume"], df["taker_buy_volume"], 168)
    if m1 is not None:
        one("1m", np.log(m1["close"]).diff(), m1["volume"], m1["taker_buy_volume"], 1440)
    return pd.DataFrame(out)
