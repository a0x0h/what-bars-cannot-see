"""How well do bar-based proxies track what they claim to measure?

Binance klines carry the aggressor-side (taker) buy volume of every bar, i.e.
the true signed order flow aggregated to the bar. This lets us score the
OHLCV-only proxies against ground truth over the full sample:

* order-flow imbalance: bulk volume classification (BVC, Easley, Lopez de
  Prado & O'Hara 2016) from hourly bars and from 1-minute bars, vs. observed
  taker imbalance;
* price impact: Kyle's lambda from BVC-signed volume vs. from true signed volume;
* volatility: Garman-Klass range estimator from hourly bars vs. realised
  variance from 1-minute returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-12


def bvc_imbalance(df: pd.DataFrame, sigma_window: int) -> pd.Series:
    r = np.log(df["close"]).diff()
    sig = r.rolling(sigma_window, min_periods=max(20, sigma_window // 4)).std()
    return pd.Series(2 * norm.cdf(r / (sig + EPS)) - 1, index=df.index)


def _pair(name, proxy, truth, level):
    d = pd.concat([proxy, truth], axis=1).dropna()
    if len(d) < 100:
        return None
    a, b = d.iloc[:, 0], d.iloc[:, 1]
    row = {"quantity": name, "aggregation": level, "n": len(d),
           "pearson": a.corr(b), "spearman": a.corr(b, method="spearman")}
    if name.startswith("order-flow"):
        nz = b != 0
        row["sign_agreement"] = float((np.sign(a[nz]) == np.sign(b[nz])).mean())
    return row


def proxy_validation(df: pd.DataFrame, X: pd.DataFrame, m1: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if "taker_buy_volume" not in df:
        return pd.DataFrame()
    v = df["volume"]
    true_sv = 2 * df["taker_buy_volume"] - v
    true_imb = true_sv / (v + EPS)
    bvc_h = bvc_imbalance(df, 168)
    for w in [1, 4, 24]:
        p = (bvc_h * v).rolling(w).sum() / (v.rolling(w).sum() + EPS)
        t = true_sv.rolling(w).sum() / (v.rolling(w).sum() + EPS)
        rows.append(_pair("order-flow imbalance: BVC(1h bars) vs taker", p, t, f"{w}h"))
    if m1 is not None and "taker_buy_volume" in m1:
        b1 = bvc_imbalance(m1, 1440) * m1["volume"]
        hr = m1.index.floor("h")
        p1 = b1.groupby(hr).sum() / (m1["volume"].groupby(hr).sum() + EPS)
        rows.append(_pair("order-flow imbalance: BVC(1m bars) vs taker", p1.reindex(df.index), true_imb, "1h"))
        tm = (2 * m1["taker_buy_volume"] - m1["volume"]) / (m1["volume"] + EPS)
        rows.append(_pair("order-flow imbalance: BVC(1m bars) vs taker", bvc_imbalance(m1, 1440), tm, "1m"))
    if {"kyle_lambda_48", "kyle_lambda_true_48"} <= set(X.columns):
        rows.append(_pair("price impact: Kyle lambda BVC vs true", X["kyle_lambda_48"], X["kyle_lambda_true_48"], "48h"))
    if {"gk_vol_24", "rv_1m_24"} <= set(X.columns):
        rows.append(_pair("volatility: log GK(1h) vs log RV(1m)", X["gk_vol_24"], X["rv_1m_24"], "24h"))
    out = pd.DataFrame([r for r in rows if r])
    return out


def proxy_validation_by_year(df, m1=None):
    v = df["volume"]
    if "taker_buy_volume" not in df:
        return pd.DataFrame()
    true_imb = (2 * df["taker_buy_volume"] - v) / (v + EPS)
    bvc_h = bvc_imbalance(df, 168)
    d = pd.concat([bvc_h.rename("bvc"), true_imb.rename("truth")], axis=1).dropna()
    return d.groupby(d.index.year).apply(lambda g: pd.Series({
        "n": len(g), "pearson": g["bvc"].corr(g["truth"]),
        "sign_agreement": (np.sign(g["bvc"]) == np.sign(g["truth"])).mean()}))


def aggtrades_check(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Sanity check: kline taker imbalance equals trade-level imbalance."""
    a = agg.copy()
    a.index = pd.to_datetime(a.index, utc=True)
    k = ((2 * df["taker_buy_volume"] - df["volume"]) / df["volume"]).rename("kline_imb")
    t = (a["signed_dollar"] / a["dollar"]).rename("trade_imb")
    d = pd.concat([k, t], axis=1).dropna()
    return pd.DataFrame([{"n_hours": len(d), "pearson": d.corr().iloc[0, 1],
                          "mean_abs_diff": (d["kline_imb"] - d["trade_imb"]).abs().mean()}])


# ----------------------------------------------------------------------------
# Extended validation: classifier variants and the information ceiling
# ----------------------------------------------------------------------------
def _flow_from_fraction(frac: pd.Series, v: pd.Series, w: int) -> pd.Series:
    sv = (2 * frac - 1) * v
    return sv.rolling(w).sum() / (v.rolling(w).sum() + EPS)


def classifier_variants(df: pd.DataFrame, m1: pd.DataFrame | None) -> pd.DataFrame:
    """Score alternative bar-based flow classifiers against observed taker imbalance.

    Variants: BVC with Normal CDF and trailing sigma windows of 24/168/720 bars,
    BVC with a Student-t(3) CDF (heavier tails, scaled to unit variance), the bar
    tick rule (all volume assigned to the sign of the bar return), and the same
    BVC / tick-rule classifiers applied to 1-minute bars and summed to the hour.
    """
    from scipy.stats import t as student_t
    if "taker_buy_volume" not in df:
        return pd.DataFrame()
    v = df["volume"]
    r = np.log(df["close"]).diff()
    true_sv = 2 * df["taker_buy_volume"] - v
    fr = {}
    for w in (24, 168, 720):
        sig = r.rolling(w, min_periods=max(20, w // 4)).std()
        fr[f"BVC Normal, sigma {w}h (1h bars)"] = pd.Series(norm.cdf(r / (sig + EPS)), index=df.index)
    sig = r.rolling(168, min_periods=48).std()
    k = np.sqrt(3 / (3 - 2))  # t(3) has variance 3, rescale to unit variance
    fr["BVC Student-t(3), sigma 168h (1h bars)"] = pd.Series(student_t.cdf(k * r / (sig + EPS), 3), index=df.index)
    fr["Tick rule (1h bars)"] = (np.sign(r) + 1) / 2
    hourly_sv = {name: (2 * f - 1) * v for name, f in fr.items()}
    if m1 is not None:
        r1 = np.log(m1["close"]).diff()
        s1 = r1.rolling(1440, min_periods=360).std()
        hr = m1.index.floor("h")
        for name, f1 in [("BVC Normal, sigma 1440m (1m bars)", pd.Series(norm.cdf(r1 / (s1 + EPS)), index=m1.index)),
                         ("Tick rule (1m bars)", (np.sign(r1) + 1) / 2)]:
            hourly_sv[name] = ((2 * f1 - 1) * m1["volume"]).groupby(hr).sum().reindex(df.index)
    rows = []
    for name, sv in hourly_sv.items():
        for w in (1, 24):
            p = sv.rolling(w).sum() / (v.rolling(w).sum() + EPS)
            t = true_sv.rolling(w).sum() / (v.rolling(w).sum() + EPS)
            d = pd.concat([p, t], axis=1).dropna()
            a, b = d.iloc[:, 0], d.iloc[:, 1]
            nz = (a != 0) & (b != 0)
            rows.append({"classifier": name, "aggregation": f"{w}h", "n": len(d), "pearson": a.corr(b),
                         "spearman": a.corr(b, method="spearman"),
                         "sign_agreement": float((np.sign(a[nz]) == np.sign(b[nz])).mean()),
                         "rmse": float(np.sqrt(((a - b) ** 2).mean()))})
    return pd.DataFrame(rows)


def information_ceiling(df: pd.DataFrame, m1: pd.DataFrame | None = None, n_bins: int = 50) -> pd.DataFrame:
    """How much of observed taker imbalance can ANY function of the bar's own return explain?

    BVC maps the standardised bar return z = r / sigma into a buy fraction, so its
    information is bounded by E[imbalance | z]. We estimate that conditional mean
    nonparametrically (quantile bins of z, fitted on the first half of the sample
    and evaluated on the second half) and compare its R^2 with that of BVC itself.
    The gap between the two is calibration that a better mapping could recover;
    the remainder 1 - ceiling is flow that leaves no trace in the bar return.
    """
    out = []

    def _one(label, r, v, taker, sig_w):
        sig = r.rolling(sig_w, min_periods=max(20, sig_w // 4)).std()
        z = r / (sig + EPS)
        imb = (2 * taker - v) / (v + EPS)
        bvc = 2 * norm.cdf(z) - 1
        d = pd.DataFrame({"z": z, "imb": imb, "bvc": bvc, "v": v}).dropna()
        d = d[d["v"] > 0]
        half = len(d) // 2
        tr, te = d.iloc[:half], d.iloc[half:]
        edges = np.unique(np.quantile(tr["z"], np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        cm = tr.groupby(pd.cut(tr["z"], edges), observed=False)["imb"].mean()
        fit = pd.cut(te["z"], edges).map(cm).astype(float)
        sst = ((te["imb"] - te["imb"].mean()) ** 2).sum()
        r2_ceiling = 1 - ((te["imb"] - fit) ** 2).sum() / sst
        # BVC as-is (no recalibration) and BVC with an OLS rescaling fitted on the first half
        r2_bvc_raw = 1 - ((te["imb"] - te["bvc"]) ** 2).sum() / sst
        b = np.polyfit(tr["bvc"], tr["imb"], 1)
        r2_bvc_lin = 1 - ((te["imb"] - np.polyval(b, te["bvc"])) ** 2).sum() / sst
        out.append({"bars": label, "n_eval": len(te), "R2_BVC_raw": r2_bvc_raw, "R2_BVC_rescaled": r2_bvc_lin,
                    "R2_ceiling_any_f(z)": r2_ceiling, "corr_BVC": te["bvc"].corr(te["imb"]),
                    "BVC_slope": b[0]})

    if "taker_buy_volume" in df:
        _one("1h", np.log(df["close"]).diff(), df["volume"], df["taker_buy_volume"], 168)
    if m1 is not None and "taker_buy_volume" in m1:
        _one("1m", np.log(m1["close"]).diff(), m1["volume"], m1["taker_buy_volume"], 1440)
    return pd.DataFrame(out)


def validation_by_regime(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """BVC accuracy (1h, 24h) by trailing-volatility tercile and by year."""
    if "taker_buy_volume" not in df:
        return pd.DataFrame()
    v = df["volume"]
    true_sv = 2 * df["taker_buy_volume"] - v
    bsv = bvc_imbalance(df, 168) * v
    pct = X["gk_vol_24"].rolling(720, min_periods=240).rank(pct=True)
    lab = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=["low vol", "mid vol", "high vol"]).astype(str)
    rows = []
    for w in (1, 24):
        p = (bsv.rolling(w).sum() / (v.rolling(w).sum() + EPS)).rename("p")
        t = (true_sv.rolling(w).sum() / (v.rolling(w).sum() + EPS)).rename("t")
        d = pd.concat([p, t], axis=1).dropna()
        zf = (d.index >= pd.Timestamp("2022-07-08", tz="UTC")) & (d.index < pd.Timestamp("2023-03-22", tz="UTC"))
        fee = pd.Series(np.where(zf, "zero-fee promo", "standard fees"), index=d.index)
        for key, grp in [("volatility", lab.reindex(d.index)), ("year", pd.Series(d.index.year.astype(str), index=d.index)),
                         ("fee_regime", fee)]:
            for g, sub in d.groupby(grp):
                if g == "nan" or len(sub) < 200:
                    continue
                rows.append({"split": key, "group": g, "aggregation": f"{w}h", "n": len(sub),
                             "pearson": sub["p"].corr(sub["t"]),
                             "sign_agreement": float((np.sign(sub["p"]) == np.sign(sub["t"])).mean())})
    return pd.DataFrame(rows)


def spread_proxy_diagnostics(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Are the low-frequency spread estimators measuring spreads or volatility?

    Binance BTC/USDT and ETH/USDT have a tick of 0.01 USDT, so the minimum
    possible quoted spread is a tiny fraction of a basis point. Reports each
    estimator's level (bps), its share of zero estimates, and its correlation
    with range-based and 1-minute realised volatility (Jahan-Parvar & Zikes 2023).
    """
    tick_bps = float((0.01 / df["close"]).median() * 1e4)
    vol_cols = [c for c in ["gk_vol_24", "rv_1m_24"] if c in X]
    rows = []
    for c, name in [("cs_spread_24", "Corwin-Schultz (24h mean)"), ("ar_spread_24", "Abdi-Ranaldo (24h)"),
                    ("roll_spread_48", "Roll (48h)")]:
        s = X[c]
        row = {"estimator": name, "median_bps": 1e4 * s.median(), "mean_bps": 1e4 * s.mean(),
               "share_zero_%": 100 * (s <= 1e-12).mean(), "one_tick_bps": tick_bps}
        ls = np.log(s.where(s > 1e-12))
        for v in vol_cols:
            row[f"corr_log_with_{v}"] = ls.corr(X[v])
        rows.append(row)
    return pd.DataFrame(rows)
