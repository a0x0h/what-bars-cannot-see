"""Causal feature construction.

Every feature at bar t is a function of bars <= t only. This is enforced by
tests/test_causality.py, which perturbs all bars after t and checks that the
features at t do not move.

Two feature sets are built:

* MS  - microstructure / liquidity proxies reconstructed from OHLCV bars,
        organised in five economically motivated groups.
* TI  - a conventional technical-indicator control set (momentum, RSI, MACD,
        Bollinger, moving-average gaps, ATR). It answers the question a referee
        will ask first: do the microstructure proxies add anything beyond
        standard price-based signals?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-12


def _rmean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).mean()


def _rstd(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=w).std()


# ----------------------------------------------------------------------------
# Spread / liquidity estimators
# ----------------------------------------------------------------------------
def corwin_schultz(h: pd.Series, l: pd.Series) -> pd.Series:
    """Corwin & Schultz (2012) two-bar high-low spread estimator (bars t-1, t)."""
    lh, ll = np.log(h), np.log(l)
    beta = (lh - ll) ** 2 + (lh.shift(1) - ll.shift(1)) ** 2
    gamma = (np.log(np.maximum(h, h.shift(1))) - np.log(np.minimum(l, l.shift(1)))) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def abdi_ranaldo(h, l, c, w: int) -> pd.Series:
    """Abdi & Ranaldo (2017) close-high-low spread, causal two-bar version.

    Original: s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]. Shifted by one bar so it
    only uses bars t-1 and t: 4 (c_{t-1} - eta_{t-1})(c_{t-1} - eta_t).
    """
    eta = (np.log(h) + np.log(l)) / 2
    lc = np.log(c)
    prod = 4 * (lc.shift(1) - eta.shift(1)) * (lc.shift(1) - eta)
    return np.sqrt(_rmean(prod, w).clip(lower=0))


def roll_spread(c, w: int) -> pd.Series:
    """Roll (1984) spread from negative serial covariance of price changes."""
    dp = np.log(c).diff()
    cov = dp.rolling(w, min_periods=w).cov(dp.shift(1))
    return 2 * np.sqrt((-cov).clip(lower=0))


# ----------------------------------------------------------------------------
# Feature builders
# ----------------------------------------------------------------------------
def build_ms_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    r = np.log(c).diff()
    dv = c * v  # dollar (quote) volume
    F = pd.DataFrame(index=df.index)

    # --- Group 1: spread and price impact -----------------------------------
    F["cs_spread_24"] = _rmean(corwin_schultz(h, l), 24)
    F["ar_spread_24"] = abdi_ranaldo(h, l, c, 24)
    F["roll_spread_48"] = roll_spread(c, 48)
    F["amihud_24"] = np.log(_rmean(r.abs() / (dv + EPS), 24) * 1e9 + EPS)

    # --- Group 2: order flow (bulk volume classification) -------------------
    sig = r.rolling(168, min_periods=48).std()
    buy_frac = pd.Series(norm.cdf(r / (sig + EPS)), index=df.index)  # Easley, LdP, O'Hara (2016)
    sv = (2 * buy_frac - 1) * v  # signed volume estimate
    vbar = _rmean(v, 168)
    F["ofi_1"] = sv / (vbar + EPS)
    ofi6 = sv.rolling(6, min_periods=6).sum() / (v.rolling(6, min_periods=6).sum() + EPS)
    ofi24 = sv.rolling(24, min_periods=24).sum() / (v.rolling(24, min_periods=24).sum() + EPS)
    F["ofi_24"] = ofi24
    F["ofi_accel"] = ofi6 - ofi24
    F["vpin_24"] = sv.abs().rolling(24, min_periods=24).sum() / (v.rolling(24, min_periods=24).sum() + EPS)
    sdv = sv * c / 1e6  # signed dollar volume, $M
    F["kyle_lambda_48"] = r.rolling(48, min_periods=48).cov(sdv) / (sdv.rolling(48, min_periods=48).var() + EPS)
    flow_cols = ["ofi_1", "ofi_24", "ofi_accel", "vpin_24", "kyle_lambda_48"]

    # --- Group 3: realised volatility and its dynamics ----------------------
    lo, lh_, ll_, lc = np.log(o), np.log(h), np.log(l), np.log(c)
    gk_bar = 0.5 * (lh_ - ll_) ** 2 - (2 * np.log(2) - 1) * (lc - lo) ** 2
    rs_bar = (lh_ - lc) * (lh_ - lo) + (ll_ - lc) * (ll_ - lo)
    gk24 = np.sqrt(_rmean(gk_bar.clip(lower=0), 24))
    gk168 = np.sqrt(_rmean(gk_bar.clip(lower=0), 168))
    F["gk_vol_24"] = np.log(gk24 + EPS)
    F["rs_vol_24"] = np.log(np.sqrt(_rmean(rs_bar.clip(lower=0), 24)) + EPS)
    n = 24
    on = lo - lc.shift(1)  # bar-to-bar "overnight" gap
    oc = lc - lo
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = _rstd(on, n) ** 2 + k * _rstd(oc, n) ** 2 + (1 - k) * _rmean(rs_bar, n)
    F["yz_vol_24"] = np.log(np.sqrt(yz.clip(lower=0)) + EPS)
    F["vol_ratio_24_168"] = np.log((gk24 + EPS) / (gk168 + EPS))
    F["vol_of_vol_168"] = _rstd(np.log(gk24 + EPS), 168)
    impact = r.abs() / np.sqrt(v / (vbar + EPS) + EPS)  # square-root impact (Bouchaud et al.)
    F["sqrt_impact_24"] = np.log(_rmean(impact, 24) + EPS)

    # --- Group 4: price efficiency and bar structure ------------------------
    path = r.abs().rolling(24, min_periods=24).sum()
    F["eff_ratio_24"] = (lc - lc.shift(24)).abs() / (path + EPS)
    F["autocorr_48"] = r.rolling(48, min_periods=48).corr(r.shift(1))
    r4 = r.rolling(4, min_periods=4).sum()
    F["var_ratio_4_168"] = _rmean(r4 ** 2, 168) / (4 * _rmean(r ** 2, 168) + EPS)
    rng = (h - l).where((h - l) > 0)
    F["body_share_24"] = (( (c - o).abs() / rng)).rolling(24, min_periods=12).mean()
    F["upper_wick_24"] = ((h - np.maximum(o, c)) / rng).rolling(24, min_periods=12).mean()
    F["lower_wick_24"] = ((np.minimum(o, c) - l) / rng).rolling(24, min_periods=12).mean()

    # --- Group 5: trading activity -----------------------------------------
    lv = np.log(v + 1)
    F["volume_z_168"] = (lv - _rmean(lv, 168)) / (_rstd(lv, 168) + EPS)
    F["volume_trend_24_168"] = np.log((_rmean(v, 24) + EPS) / (vbar + EPS))

    groups = {
        "spread_impact": ["cs_spread_24", "ar_spread_24", "roll_spread_48", "amihud_24"],
        "order_flow": flow_cols,
        "volatility": ["gk_vol_24", "rs_vol_24", "yz_vol_24", "vol_ratio_24_168", "vol_of_vol_168", "sqrt_impact_24"],
        "efficiency_structure": ["eff_ratio_24", "autocorr_48", "var_ratio_4_168", "body_share_24",
                                 "upper_wick_24", "lower_wick_24"],
        "activity": ["volume_z_168", "volume_trend_24_168"],
    }
    return F, groups


def build_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """Observed aggressor-side flow and trade counts (Binance taker-buy volume).

    These are the 'ground truth' counterparts of the bar-based proxies above.
    Returns an empty frame when the data source has no such columns.
    """
    T = pd.DataFrame(index=df.index)
    v, c = df["volume"], df["close"]
    r = np.log(c).diff()
    if "taker_buy_volume" in df and df["taker_buy_volume"].notna().mean() > 0.95:
        tsv = 2 * df["taker_buy_volume"] - v
        vs24 = v.rolling(24, min_periods=24).sum()
        T["taker_imb_1"] = tsv / (v + EPS)
        t6 = tsv.rolling(6, min_periods=6).sum() / (v.rolling(6, min_periods=6).sum() + EPS)
        T["taker_imb_24"] = tsv.rolling(24, min_periods=24).sum() / (vs24 + EPS)
        T["taker_imb_accel"] = t6 - T["taker_imb_24"]
        T["taker_vpin_24"] = tsv.abs().rolling(24, min_periods=24).sum() / (vs24 + EPS)
        sdv = tsv * c / 1e6
        T["kyle_lambda_true_48"] = r.rolling(48, min_periods=48).cov(sdv) / (sdv.rolling(48, min_periods=48).var() + EPS)
    if "trades" in df and df["trades"].notna().mean() > 0.95 and (df["trades"] > 0).mean() > 0.95:
        n = df["trades"].clip(lower=1)
        ln = np.log(n)
        T["trades_z_168"] = (ln - _rmean(ln, 168)) / (_rstd(ln, 168) + EPS)
        size = np.log(v * c / n + EPS)  # average dollar trade size
        T["trade_size_z_168"] = (size - _rmean(size, 168)) / (_rstd(size, 168) + EPS)
    return T


def build_intrahour_features(m1: pd.DataFrame, hour_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-level realised measures from 1-minute bars of the same hour (all
    minutes of hour t close by the close of bar t, so this is causal).

    rv   realised variance; bv bipower variation (Barndorff-Nielsen & Shephard
    2004); jump share max(rv - bv, 0)/rv; realised skewness (Amaya et al. 2015).
    """
    lc = np.log(m1["close"])
    r = lc.diff()
    hour = m1.index.floor("h")
    g = pd.DataFrame({"r2": r ** 2, "r3": r ** 3, "bvp": r.abs() * r.abs().shift(1), "hour": hour,
                      "n": r.notna().astype(int)})
    if "taker_buy_volume" in m1:
        tsv = 2 * m1["taker_buy_volume"] - m1["volume"]
        late = m1.index.minute >= 45
        g["tsv_late"] = tsv.where(late, 0.0)
        g["v_late"] = m1["volume"].where(late, 0.0)
    agg = g.groupby("hour").sum(min_count=1)
    agg = agg[agg["n"] >= 50]  # require most minutes present
    rv = agg["r2"]
    bv = (np.pi / 2) * agg["bvp"]
    H = pd.DataFrame(index=agg.index)
    rv24 = rv.reindex(hour_index).rolling(24, min_periods=20).sum()
    H["rv_1m_24"] = np.log(rv24 + EPS)
    H["jump_share_24"] = ((rv - bv).clip(lower=0) / (rv + EPS)).reindex(hour_index).rolling(24, min_periods=20).mean()
    rskew = np.sqrt(agg["n"]) * agg["r3"] / (rv ** 1.5 + EPS)
    H["rskew_24"] = rskew.reindex(hour_index).rolling(24, min_periods=20).mean()
    H["rv_1m_1"] = np.log(rv.reindex(hour_index) + EPS)
    if "tsv_late" in agg:
        H["taker_imb_late15"] = (agg["tsv_late"] / (agg["v_late"] + EPS)).reindex(hour_index)
    return H.reindex(hour_index)


def build_ti_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    lc = np.log(c)
    T = pd.DataFrame(index=df.index)
    for k in [1, 4, 24, 168]:
        T[f"mom_{k}"] = lc - lc.shift(k)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    T["rsi_14"] = 100 - 100 / (1 + up / (dn + EPS))
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    T["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    sma20, sd20 = _rmean(c, 20), _rstd(c, 20)
    T["bb_pctb_20"] = (c - (sma20 - 2 * sd20)) / (4 * sd20 + EPS)
    T["ma_gap_24"] = lc - np.log(_rmean(c, 24))
    T["ma_gap_168"] = lc - np.log(_rmean(c, 168))
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    T["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    return T


def build_targets(df: pd.DataFrame, horizons) -> pd.DataFrame:
    """y_h(t) = log C_{t+h} - log C_t  (decision at the close of bar t)."""
    lc = np.log(df["close"])
    Y = pd.DataFrame(index=df.index)
    for hz in horizons:
        Y[f"y_{hz}"] = lc.shift(-hz) - lc
    Y["r1_next"] = df["close"].shift(-1) / df["close"] - 1  # simple return over (t, t+1], for P&L
    return Y


def build_dataset(df: pd.DataFrame, horizons=(1, 4, 8, 24), m1: pd.DataFrame | None = None):
    """Return (X, Y, feature_sets, groups) with warm-up rows removed.

    Feature sets
      TI        technical-indicator controls
      MS_proxy  microstructure proxies computable from plain OHLCV bars
      MS        MS_proxy + observed taker flow / trade counts + 1-minute realised measures
      ALL       MS + TI
    """
    ms, groups = build_ms_features(df)
    tr = build_trade_features(df)
    blocks = [ms]
    if not tr.empty:
        groups["trade_level"] = list(tr.columns)
        blocks.append(tr)
    if m1 is not None:
        ih = build_intrahour_features(m1, df.index)
        late = [c for c in ih.columns if c == "taker_imb_late15"]
        groups["intrahour_1m"] = [c for c in ih.columns if c not in late]
        if late:
            groups.setdefault("trade_level", []).extend(late)
        blocks.append(ih)
    ti = build_ti_features(df)
    X = pd.concat(blocks + [ti], axis=1).replace([np.inf, -np.inf], np.nan)
    Y = build_targets(df, horizons)
    valid = X.notna().all(axis=1)
    first = valid.idxmax()  # end of warm-up
    X, Y = X.loc[first:], Y.loc[first:]
    X = X[X.notna().all(axis=1)]  # rows inside data gaps
    Y = Y.loc[X.index]
    proxy = list(ms.columns)
    full = [c for b in blocks for c in b.columns]
    sets = {"TI": list(ti.columns), "MS_proxy": proxy, "MS": full, "ALL": full + list(ti.columns)}
    # Matched flow-only sets: the same five constructions from BVC-signed and from observed taker volume
    true_flow = ["taker_imb_1", "taker_imb_24", "taker_imb_accel", "taker_vpin_24", "kyle_lambda_true_48"]
    if all(c in X.columns for c in true_flow):
        sets["FLOW_proxy"] = list(groups["order_flow"])
        sets["FLOW_true"] = true_flow
    return X, Y, sets, groups
