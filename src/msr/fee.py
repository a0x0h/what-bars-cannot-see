"""Binance's zero-fee promotion on BTC pairs as a natural experiment in what bars measure.

From 14:00 UTC on 8 July 2022 to 00:00 UTC on 22 March 2023 Binance charged no maker or taker fee
on BTC/USDT and twelve other BTC pairs; USDT pairs of other coins kept standard fees. We treat
BTC/USDT as the treated unit and a panel of USDT pairs as controls, compute daily measures from
one-minute bars, and estimate the change in BTC relative to the controls when fees were switched
off (the "on" event of the promotion) and back on (the "off" event).

Daily measures (all from one-minute bars of the day, dollar amounts in USDT):
    trades, dvol          number of trades, dollar volume
    size                  average trade size dvol / trades
    net1m                 directional volume sum_m |B_m - S_m| (B, S taker buy and sell dollar volume)
    two_sided             dvol - net1m: volume that is matched by opposite aggressor flow in the
                          same minute; the part of volume with no net direction
    ts_share              two_sided / dvol
    abs_imb_h             mean over the day's hours of |hourly taker imbalance|
    rv                    realised variance, sum of squared one-minute log returns
    amihud                mean over the day's hours of |r_h| / dollar volume_h (x 1e9)
    lam_true, r2_true     slope (bps per $1M) and R^2 of one-minute returns on observed signed dollar flow
    lam_bvc, r2_bvc       the same with BVC-signed one-minute dollar volume
    corr_bvc              correlation of one-minute BVC imbalance with observed one-minute imbalance
    vpin_true, vpin_bvc   sum_h |signed volume_h| / sum_h volume_h with observed and with hourly-bar BVC
    ac1                   first-order autocorrelation of one-minute returns

Inference. With one treated unit, standard panel standard errors are not valid (Conley and Taber,
2011). We report (i) Newey-West t-statistics for the mean shift of the daily difference between
BTC and the equal-weighted control average, (ii) a permutation p-value from in-space placebos in
which each control in turn is treated as if it had been treated, with the other controls as its
control group (Abadie, Diamond and Hainmueller, 2010), and (iii) the rank of the actual 90-day jump
among 90-day jumps at placebo dates in 2024-2026, when no fee change affected BTC/USDT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from .events import TUSD_OFF, ZERO_FEE_OFF, ZERO_FEE_ON

EPS = 1e-12

# Estimation windows (whole UTC days). The switch days themselves are left out.
PRE = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-07-08", tz="UTC"))
ZERO = (pd.Timestamp("2022-07-09", tz="UTC"), pd.Timestamp("2023-03-22", tz="UTC"))
POST = (pd.Timestamp("2023-03-22", tz="UTC"), TUSD_OFF.normalize())  # ends before the next fee change on BTC pairs
PLACEBO = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))

# name: (column, transform, label)
OUTCOMES = {
    "trades": ("trades", "log", "Number of trades"),
    "dvol": ("dvol", "log", "Dollar volume"),
    "size": ("size", "log", "Average trade size"),
    "net1m": ("net1m", "log", "Directional volume"),
    "two_sided": ("two_sided", "log", "Two-sided volume"),
    "ts_share": ("ts_share", "level", "Two-sided share of volume"),
    "abs_imb_h": ("abs_imb_h", "log", "Mean |hourly imbalance|"),
    "rv": ("rv", "log", "Realised variance"),
    "amihud": ("amihud", "log", "Amihud illiquidity (hourly bars)"),
    "lam_bvc": ("lam_bvc", "log", "Kyle's lambda, BVC flow"),
    "lam_true": ("lam_true", "log", "Kyle's lambda, observed flow"),
    "r2_true": ("r2_true", "level", "$R^2$ of returns on observed flow"),
    "corr_bvc": ("corr_bvc", "level", "corr(BVC, observed imbalance)"),
    "vpin_true": ("vpin_true", "log", "VPIN, observed flow"),
    "vpin_bvc": ("vpin_bvc", "log", "VPIN, BVC"),
    "ac1": ("ac1", "level", "One-minute return autocorrelation"),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_klines(path: str | Path, freq: str) -> pd.DataFrame:
    """Binance kline CSV written by scripts/download_binance.py, on a complete UTC grid."""
    cols = ["open_time", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume"]
    d = pd.read_csv(path, usecols=cols)
    d.index = pd.to_datetime(d.pop("open_time"), unit="ms", utc=True)
    d = d[~d.index.duplicated(keep="first")].sort_index()
    d = d[(d["close"] > 0) & (d["high"] >= d["low"])]
    return d.reindex(pd.date_range(d.index[0], d.index[-1], freq=freq, tz="UTC"))


def _ols_from_sums(n, sx, sy, sxx, syy, sxy):
    cxy = sxy - sx * sy / n
    vx = sxx - sx ** 2 / n
    vy = syy - sy ** 2 / n
    slope = cxy / vx.where(vx > 0)
    r2 = cxy ** 2 / (vx * vy).where((vx > 0) & (vy > 0))
    return slope, r2


def daily_measures(m1: pd.DataFrame, min_minutes: int = 1200) -> pd.DataFrame:
    """Daily measures from one-minute klines (see module docstring)."""
    c, qv = m1["close"], m1["quote_volume"]
    r = np.log(c).diff() * 1e4                                   # bps; NaN after a missing minute
    q = (2 * m1["taker_buy_quote_volume"] - qv)                  # observed signed dollar flow
    sig = r.rolling(1440, min_periods=360).std()
    bfrac = pd.Series(norm.cdf(r / (sig + EPS)), index=m1.index).where(r.notna() & sig.notna())
    b = (2 * bfrac - 1) * qv                                     # BVC signed dollar flow
    day = m1.index.floor("D")

    # A minute counts if both its close and the previous close exist. Minutes without trades (zero
    # volume, unchanged close) are genuine observations of no flow and no price change and are kept in
    # the regressions; imbalances need positive volume.
    ok = r.notna() & q.notna()
    x, y = (q / 1e6).where(ok), r.where(ok)                      # $M, bps
    okb = ok & b.notna()
    xb, yb = (b / 1e6).where(okb), r.where(okb)
    oki = okb & (qv > 0)
    imb_t = (q / qv).where(oki)
    imb_b = (2 * bfrac - 1).where(oki)
    r_lag = r.shift(1)
    oka = r.notna() & r_lag.notna()

    g = pd.DataFrame({
        "n": ok.astype(float), "trades": m1["trades"], "dvol": qv, "net1m": q.abs(), "rv": (r / 1e4) ** 2,
        "sx": x, "sy": y, "sxx": x ** 2, "syy": y ** 2, "sxy": x * y,
        "nb": okb.astype(float), "bx": xb, "by": yb, "bxx": xb ** 2, "byy": yb ** 2, "bxy": xb * yb,
        "ni": oki.astype(float), "ix": imb_b, "iy": imb_t, "ixx": imb_b ** 2, "iyy": imb_t ** 2, "ixy": imb_b * imb_t,
        "na": oka.astype(float), "ax": r_lag.where(oka), "ay": r.where(oka), "axx": r_lag.where(oka) ** 2,
        "ayy": r.where(oka) ** 2, "axy": (r_lag * r).where(oka),
    }).groupby(day).sum(min_count=1)

    D = pd.DataFrame(index=g.index)
    D["n_minutes"] = g["n"]
    D["trades"], D["dvol"] = g["trades"], g["dvol"]
    D["size"] = g["dvol"] / g["trades"]
    D["net1m"] = g["net1m"]
    D["two_sided"] = (g["dvol"] - g["net1m"]).clip(lower=0)
    D["ts_share"] = D["two_sided"] / g["dvol"]
    D["rv"] = g["rv"]
    D["lam_true"], D["r2_true"] = _ols_from_sums(g["n"], g["sx"], g["sy"], g["sxx"], g["syy"], g["sxy"])
    D["lam_bvc"], D["r2_bvc"] = _ols_from_sums(g["nb"], g["bx"], g["by"], g["bxx"], g["byy"], g["bxy"])
    _, r2i = _ols_from_sums(g["ni"], g["ix"], g["iy"], g["ixx"], g["iyy"], g["ixy"])
    cxy = g["ixy"] - g["ix"] * g["iy"] / g["ni"]
    D["corr_bvc"] = np.sign(cxy) * np.sqrt(r2i)
    ca = g["axy"] - g["ax"] * g["ay"] / g["na"]
    D["ac1"] = ca / np.sqrt((g["axx"] - g["ax"] ** 2 / g["na"]) * (g["ayy"] - g["ay"] ** 2 / g["na"]))

    # Hourly bars built from the minutes: Amihud, hourly imbalance, VPIN with observed flow and BVC
    hour = m1.index.floor("h")
    H = pd.DataFrame({"r": r / 1e4, "q": q, "qv": qv, "n": ok.astype(float)}).groupby(hour).sum(min_count=1)
    H = H[H["n"] >= 50]
    H = H.reindex(pd.date_range(H.index[0], H.index[-1], freq="h", tz="UTC"))
    sig_h = H["r"].rolling(168, min_periods=48).std()
    H["b"] = (2 * norm.cdf(H["r"] / (sig_h + EPS)) - 1) * H["qv"]
    H.loc[sig_h.isna() | H["r"].isna(), "b"] = np.nan
    H["amihud"] = H["r"].abs() / H["qv"].where(H["qv"] > 0) * 1e9
    H["abs_imb"] = (H["q"] / H["qv"].where(H["qv"] > 0)).abs()
    hd = H.index.floor("D")
    Hd = pd.DataFrame({"amihud": H["amihud"], "abs_imb_h": H["abs_imb"], "aq": H["q"].abs(),
                       "ab": H["b"].abs(), "qv": H["qv"], "qv_b": H["qv"].where(H["b"].notna())})
    agg = Hd.groupby(hd).agg(amihud=("amihud", "mean"), abs_imb_h=("abs_imb_h", "mean"), aq=("aq", "sum"),
                             ab=("ab", "sum"), qv=("qv", "sum"), qv_b=("qv_b", "sum"))
    D["amihud"] = agg["amihud"]
    D["abs_imb_h"] = agg["abs_imb_h"]
    D["vpin_true"] = agg["aq"] / agg["qv"]
    D["vpin_bvc"] = agg["ab"] / agg["qv_b"].where(agg["qv_b"] > 0)
    return D[D["n_minutes"] >= min_minutes]


def hourly_measures(h1: pd.DataFrame) -> pd.DataFrame:
    """Hourly trades, dollar volume and |imbalance| for the switch-window plots."""
    qv = h1["quote_volume"]
    return pd.DataFrame({"trades": h1["trades"], "dvol": qv,
                         "abs_imb": ((2 * h1["taker_buy_quote_volume"] - qv) / qv.where(qv > 0)).abs()},
                        index=h1.index)


# ----------------------------------------------------------------------------
# Difference-in-differences with one treated unit
# ----------------------------------------------------------------------------
def _transform(s: pd.Series, how: str) -> pd.Series:
    if how == "log":
        return np.log(s.where(s > 0))
    return s


def outcome_matrix(panel: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    col, how, _ = OUTCOMES[key]
    return pd.DataFrame({s: _transform(d[col], how) for s, d in panel.items()})


def diff_series(Y: pd.DataFrame, treated: str, controls: list[str], min_share: float = 0.75) -> pd.Series:
    """Treated outcome minus the equal-weighted average of the controls available that day."""
    C = Y[controls]
    avg = C.mean(axis=1).where(C.notna().mean(axis=1) >= min_share)
    return (Y[treated] - avg).dropna()


def _in(idx, win):
    return (idx >= win[0]) & (idx < win[1])


def did_estimate(D: pd.Series, hac_lags: int = 30, windows=None, trend: bool = False) -> dict:
    """Mean shift of the daily difference D: zero-fee window vs. before, and after vs. zero-fee window.
    With trend=True a linear time trend in D is added (identified from the windows before and after)."""
    w_pre, w_zero, w_post = windows or (PRE, ZERO, POST)
    pre, zero, post = _in(D.index, w_pre), _in(D.index, w_zero), _in(D.index, w_post)
    d = D[pre | zero | post]
    Z = pd.DataFrame({"const": 1.0, "zero": zero[pre | zero | post].astype(float),
                      "post": post[pre | zero | post].astype(float)}, index=d.index)
    if trend:
        Z["trend"] = (d.index - d.index[0]).days / 365.0
    res = sm.OLS(d.values, Z.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    b, V = res.params, res.cov_params()
    on, on_se = b[1], np.sqrt(V[1, 1])
    off = b[2] - b[1]
    off_se = np.sqrt(V[1, 1] + V[2, 2] - 2 * V[1, 2])
    return {"on": on, "on_se": on_se, "on_t": on / on_se, "on_p": 2 * norm.sf(abs(on / on_se)),
            "off": off, "off_se": off_se, "off_t": off / off_se, "off_p": 2 * norm.sf(abs(off / off_se)),
            "post_vs_pre": b[2], "n_pre": int(pre.sum()), "n_zero": int(zero.sum()), "n_post": int(post.sum())}


def local_did(D: pd.Series, switch: pd.Timestamp, days: int = 90, hac_lags: int = 30) -> dict:
    """Event-window difference-in-differences: mean of D over the `days` whole days after a switch
    minus the mean over the `days` whole days before it, with a Newey-West t-statistic. A day that
    straddles the switch (8 July 2022) belongs to neither window."""
    start = D.index
    day0 = switch.normalize()
    first_after = day0 if switch == day0 else day0 + pd.Timedelta(days=1)
    before = (start >= day0 - pd.Timedelta(days=days)) & (start < day0)
    after = (start >= first_after) & (start < first_after + pd.Timedelta(days=days))
    d = D[before | after]
    if before.sum() < 0.8 * days or after.sum() < 0.8 * days:
        return {"b": np.nan, "t": np.nan, "n": 0}
    X = np.column_stack([np.ones(len(d)), after[before | after].astype(float)])
    res = sm.OLS(d.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {"b": float(res.params[1]), "t": float(res.tvalues[1]), "n": int(len(d))}


def jump(D: pd.Series, switch: pd.Timestamp, days: int = 90) -> float:
    return local_did(D, switch, days)["b"]


def did_table(panel: dict[str, pd.DataFrame], treated: str, controls: list[str], keys=None,
              hac_lags: int = 30, days: int = 90, windows=None) -> pd.DataFrame:
    """Effects of removing and restoring fees.

    Main estimate: event-window difference-in-differences over `days` days on each side of each
    switch (local_did) with three kinds of inference: Newey-West t, in-space placebo p-value (each
    control in turn treated as if it had been treated) and in-time placebo p-value (the same statistic
    at weekly placebo dates in 2024-2026). Robustness: the full-window regression (did_estimate), with
    and without a linear trend in the treated-minus-control difference."""
    keys = keys or list(OUTCOMES)
    switches = {"on": ZERO_FEE_ON, "off": ZERO_FEE_OFF}
    plac_dates = pd.date_range(PLACEBO[0] + pd.Timedelta(days=days), PLACEBO[1] - pd.Timedelta(days=days), freq="7D")
    rows = []
    for k in keys:
        Y = outcome_matrix(panel, k)
        D = diff_series(Y, treated, controls)
        row = {"outcome": k, "label": OUTCOMES[k][2], "transform": OUTCOMES[k][1]}
        pj = np.array([jump(D, t, days) for t in plac_dates])
        pj = np.abs(pj[~np.isnan(pj)])
        row["n_time_placebos"] = len(pj)
        for ev, sw in switches.items():
            e = local_did(D, sw, days, hac_lags)
            row[ev], row[f"{ev}_t"] = e["b"], e["t"]
            row[f"{ev}_p"] = 2 * norm.sf(abs(e["t"])) if np.isfinite(e["t"]) else np.nan
            plac = []
            if len(controls) >= 3:
                for c in controls:
                    plac.append(jump(diff_series(Y, c, [o for o in controls if o != c]), sw, days))
            plac = np.abs(np.array([x for x in plac if np.isfinite(x)]))
            row[f"p_space_{ev}"] = (1 + np.sum(plac >= abs(e["b"]))) / (1 + len(plac)) if len(plac) else np.nan
            row[f"placebo_{ev}_abs_max"] = plac.max() if len(plac) else np.nan
            row[f"p_time_{ev}"] = (1 + np.sum(pj >= abs(e["b"]))) / (1 + len(pj)) if len(pj) else np.nan
        full = did_estimate(D, hac_lags, windows)
        trd = did_estimate(D, hac_lags, windows, trend=True)
        for ev in ("on", "off"):
            row[f"{ev}_full"], row[f"{ev}_full_t"] = full[ev], full[f"{ev}_t"]
            row[f"{ev}_trend"], row[f"{ev}_trend_t"] = trd[ev], trd[f"{ev}_t"]
        # Treated-unit levels in each window (medians of the raw daily measure)
        raw = panel[treated][OUTCOMES[k][0]]
        row.update({f"btc_{w}": raw[_in(raw.index, win)].median() for w, win in [("pre", PRE), ("zero", ZERO),
                                                                                  ("post", POST)]})
        rows.append(row)
    return pd.DataFrame(rows)


def event_study(panel: dict[str, pd.DataFrame], treated: str, controls: list[str], keys,
                start="2022-01-01", end="2023-12-31") -> pd.DataFrame:
    """Weekly means of the treated-minus-controls difference, relative to its mean over the 90 days
    before the promotion (the baseline of the event-window estimate)."""
    out = {}
    for k in keys:
        D = diff_series(outcome_matrix(panel, k), treated, controls)
        base = D[(D.index >= ZERO_FEE_ON.normalize() - pd.Timedelta(days=90)) & (D.index < ZERO_FEE_ON.normalize())].mean()
        s = (D - base)[(D.index >= pd.Timestamp(start, tz="UTC")) & (D.index <= pd.Timestamp(end, tz="UTC"))]
        out[k] = s.resample("W-MON", label="left", closed="left").mean()
    return pd.DataFrame(out)


def switch_window(hourly: dict[str, pd.DataFrame], treated: str, controls: list[str], event: pd.Timestamp,
                  hours: int = 72) -> pd.DataFrame:
    """Hourly treated and average-control measures around a fee switch, each relative to its mean over
    the `hours` before the switch (logs for trades and volume, levels for |imbalance|)."""
    lo, hi = event - pd.Timedelta(hours=hours), event + pd.Timedelta(hours=hours)
    rows = {}
    for k in ["trades", "dvol", "abs_imb"]:
        M = pd.DataFrame({s: h[k] for s, h in hourly.items()}).loc[lo:hi - pd.Timedelta(hours=1)]
        if k != "abs_imb":
            M = np.log(M.where(M > 0))
        base = M[M.index < event].mean()
        rel = M - base
        rows[f"{k}_treated"] = rel[treated]
        rows[f"{k}_controls"] = rel[controls].mean(axis=1)
        # level of |imbalance| itself is more readable than a difference (6-hour moving average)
        if k == "abs_imb":
            rows[f"{k}_treated_level"] = M[treated].rolling(6, min_periods=3).mean()
            rows[f"{k}_controls_level"] = M[controls].mean(axis=1).rolling(6, min_periods=3).mean()
    out = pd.DataFrame(rows)
    out.index = (out.index - event) / pd.Timedelta(hours=1)
    out.index.name = "hours_from_switch"
    return out


def hourly_diff(hourly: dict[str, pd.DataFrame], treated: str, controls: list[str], key: str) -> pd.Series:
    """Hourly log treated measure minus the mean log measure of the controls."""
    M = pd.DataFrame({s: h[key] for s, h in hourly.items()})
    M = np.log(M.where(M > 0))
    return M[treated] - M[controls].mean(axis=1)


def hour_jump(D: pd.Series, event: pd.Timestamp, hours: int = 24) -> float:
    b = D[(D.index >= event - pd.Timedelta(hours=hours)) & (D.index < event)]
    a = D[(D.index >= event) & (D.index < event + pd.Timedelta(hours=hours))]
    if a.notna().sum() < 0.8 * hours or b.notna().sum() < 0.8 * hours:
        return np.nan
    return a.mean() - b.mean()


def switch_jumps(hourly: dict[str, pd.DataFrame], treated: str, controls: list[str], event: pd.Timestamp,
                 hours: int = 24) -> dict:
    """Change in the treated-minus-controls difference over `hours` after vs. before a switch (logs)."""
    return {k: hour_jump(hourly_diff(hourly, treated, controls, k), event, hours)
            for k in ["trades", "dvol", "abs_imb"]}


# ----------------------------------------------------------------------------
# Where the fee-free volume went: other BTC pairs
# ----------------------------------------------------------------------------
def pair_daily(pairs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily dollar volume, trades and volume-weighted |hourly imbalance| for each BTC pair.

    For non-USD quote currencies the quote volume is in that currency; all pairs used here are
    quoted in dollar stablecoins, so quote volume is treated as dollars."""
    rows = []
    for name, h in pairs.items():
        qv = h["quote_volume"]
        q = 2 * h["taker_buy_quote_volume"] - qv
        day = h.index.floor("D")
        g = pd.DataFrame({"dvol": qv, "trades": h["trades"], "aq": q.abs()}).groupby(day).sum(min_count=1)
        g["vw_abs_imb"] = g["aq"] / g["dvol"].where(g["dvol"] > 0)
        g["pair"] = name
        rows.append(g.reset_index(names="day"))
    return pd.concat(rows, ignore_index=True)


def pair_regime_table(pd_daily: pd.DataFrame) -> pd.DataFrame:
    """Median daily dollar volume ($M) and mean volume-weighted |imbalance| by pair and fee window."""
    wins = {"Jan-Jul 2022 (fees)": (PRE[0], ZERO_FEE_ON.normalize()),
            "Jul 2022-Mar 2023 (13 pairs free)": (ZERO[0], ZERO[1]),
            "Mar-Sep 2023 (only TUSD free)": (POST[0], POST[1]),
            "Sep-Dec 2023": (TUSD_OFF.normalize() + pd.Timedelta(days=1), pd.Timestamp("2024-01-01", tz="UTC"))}
    rows = []
    for pair, g in pd_daily.groupby("pair"):
        for w, (a, b) in wins.items():
            s = g[(g["day"] >= a) & (g["day"] < b)]
            rows.append({"pair": pair, "window": w, "days": len(s),
                         "median_dvol_musd": s["dvol"].median() / 1e6 if len(s) else np.nan,
                         "mean_vw_abs_imb": s["vw_abs_imb"].mean() if len(s) else np.nan,
                         "median_trades_k": s["trades"].median() / 1e3 if len(s) else np.nan})
    return pd.DataFrame(rows)
