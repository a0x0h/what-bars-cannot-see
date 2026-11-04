"""External validity: the measurement and flow-response results on 20 Binance USDT pairs.

    python run_panel.py          # needs data/raw/panel (1h + 1m klines, see scripts/download_binance.py)

For every coin: BVC accuracy, the information ceiling (in sample and out of sample, hourly and
one-minute bars), BVC calibration slope, Kyle's lambda from BVC vs. from observed flow, range-based
vs. realised volatility, the Corwin-Schultz spread against the tick, the return response to an
observed-flow shock, and the single-predictor out-of-sample test. Output: results/PANEL/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from msr import analysis, robust, validation  # noqa: E402
from msr.data import load_ohlcv  # noqa: E402
from msr.events import zero_fee_mask  # noqa: E402
from msr.features import build_dataset  # noqa: E402
from msr.single import single_predictor_oos  # noqa: E402

PANEL = ROOT / "data" / "raw" / "panel"
OUT = ROOT / "results" / "PANEL"


def tick_bps(m1: pd.DataFrame) -> float:
    """Relative tick: smallest positive one-minute close change in each month over the month's median
    price, median across months (Binance tick sizes are fixed per pair but change occasionally)."""
    c = m1["close"].dropna()
    c.index = c.index.tz_localize(None)
    dc = c.diff().abs()
    dc = dc[dc > 0]
    tick = dc.groupby(dc.index.to_period("M")).min()
    px = c.groupby(c.index.to_period("M")).median()
    return float((tick / px.reindex(tick.index) * 1e4).median())


def one_asset(sym: str, cfg: dict) -> dict:
    t0 = time.time()
    df = load_ohlcv(PANEL / f"{sym}_1h.csv", verbose=False)
    m1 = load_ohlcv(PANEL / f"{sym}_1m.csv.gz", verbose=False, freq="1min")
    X, Y, sets, groups = build_dataset(df, (1, 2, 4), m1=m1)
    row = {"symbol": sym, "hours": len(X), "median_dvol_musd": float((df["close"] * df["volume"]).median() / 1e6)}

    val = validation.proxy_validation(df, X, m1).set_index(["quantity", "aggregation"])
    row["corr_bvc_1h"] = val.loc[("order-flow imbalance: BVC(1h bars) vs taker", "1h"), "pearson"]
    row["sign_bvc_1h"] = val.loc[("order-flow imbalance: BVC(1h bars) vs taker", "1h"), "sign_agreement"]
    row["corr_bvc_24h"] = val.loc[("order-flow imbalance: BVC(1h bars) vs taker", "24h"), "pearson"]
    row["corr_bvc1m_1h"] = val.loc[("order-flow imbalance: BVC(1m bars) vs taker", "1h"), "pearson"]
    row["corr_lambda"] = val.loc[("price impact: Kyle lambda BVC vs true", "48h"), "pearson"]
    row["corr_gk_rv"] = val.loc[("volatility: log GK(1h) vs log RV(1m)", "24h"), "pearson"]

    ceil = validation.information_ceiling(df, m1).set_index("bars")
    row["ceil_oos_1h"] = ceil.loc["1h", "R2_ceiling_any_f(z)"]
    row["ceil_oos_1m"] = ceil.loc["1m", "R2_ceiling_any_f(z)"]
    cis = robust.ceiling_in_sample(df, m1)
    full = cis[cis["period"] == "full sample"].set_index("bars")
    row["ceil_is_1h"] = full.loc["1h", "R2_ceiling_in_sample"]
    row["corr2_bvc_1h"] = full.loc["1h", "corr2_BVC"]
    row["ceil_is_1m"] = full.loc["1m", "R2_ceiling_in_sample"]
    row["slope_imb_on_bvc_1h"] = full.loc["1h", "slope_imb_on_BVC"]
    row["sd_imb_1h"] = full.loc["1h", "sd_imb"]
    # yearly in-sample ceilings (stability)
    yr = cis[(cis["bars"] == "1h") & cis["period"].str.fullmatch(r"\d{4}")]
    row["ceil_is_1h_min_year"] = yr["R2_ceiling_in_sample"].min()
    row["ceil_is_1h_max_year"] = yr["R2_ceiling_in_sample"].max()

    row["tick_bps"] = tick_bps(m1)
    cs = X["cs_spread_24"]
    row["cs_median_bps"] = float(1e4 * cs.median())
    row["cs_corr_gk"] = float(np.log(cs.where(cs > 1e-12)).corr(X["gk_vol_24"]))

    fr = analysis.flow_response(df, X, horizons=(1, 2, 4))
    for k in (1, 2, 4):
        s = fr[(fr.k == k) & (fr.spec == "taker")].iloc[0]
        row[f"fr_taker_b{k}"], row[f"fr_taker_t{k}"] = s["beta_bps"], s["t"]
        s = fr[(fr.k == k) & (fr.spec == "bvc")].iloc[0]
        row[f"fr_bvc_b{k}"], row[f"fr_bvc_t{k}"] = s["beta_bps"], s["t"]
        s = fr[(fr.k == k) & (fr.spec == "joint") & (fr.regressor == "taker")].iloc[0]
        row[f"fr_joint_b{k}"], row[f"fr_joint_t{k}"] = s["beta_bps"], s["t"]
    # the same response outside BTC's zero-fee window (relevant for BTC only)
    keep = ~zero_fee_mask(X.index)
    s = analysis.flow_response(df, X[keep], horizons=(2,))
    s = s[(s.spec == "taker")].iloc[0]
    row["fr_taker_b2_stdfees"], row["fr_taker_t2_stdfees"] = s["beta_bps"], s["t"]

    wf = cfg["walkforward"]
    for name, col in [("taker", "taker_imb_1"), ("bvc", "ofi_1")]:
        sp = single_predictor_oos(X[col], df, horizons=(2,), min_train=wf["min_train_hours"], step=wf["step_hours"])
        row[f"sp_{name}_r2oos_2"] = float(sp["R2oos_%"].iloc[0])
        row[f"sp_{name}_cwp_2"] = float(sp["CW_p"].iloc[0])
    print(f"  {sym}: corr {row['corr_bvc_1h']:.2f} ceiling {row['ceil_is_1h']:.2f} "
          f"fr2 {row['fr_taker_b2']:.2f} (t {row['fr_taker_t2']:.1f}) [{time.time() - t0:.0f}s]", flush=True)
    lc = np.log(df["close"])
    x = X["taker_imb_1"]
    ser = pd.DataFrame({"x": (x - x.mean()) / x.std()}, index=X.index)
    for k in (1, 2, 4):
        ser[f"y{k}"] = ((lc.shift(-k) - lc) * 1e4).reindex(X.index)
    return row, ser


def pooled_dk(series: dict[str, pd.DataFrame], k: int, time_fe: bool = False) -> dict:
    """Pooled regression of the k-hour return (bps) on standardised observed imbalance with coin fixed
    effects, and Driscoll-Kraay standard errors (Newey-West on the hourly cross-sectional sums of
    scores), which allow for any correlation across coins and over time. With time_fe the hourly
    cross-sectional means are removed first, so the coefficient uses only relative flow and returns."""
    S = pd.concat({s: d[["x", f"y{k}"]] for s, d in series.items()}, names=["coin", "time"]).dropna()
    S = S.rename(columns={f"y{k}": "y"})
    S["y"] = S["y"] - S.groupby(level="coin")["y"].transform("mean")
    S["x"] = S["x"] - S.groupby(level="coin")["x"].transform("mean")
    if time_fe:
        S["y"] = S["y"] - S.groupby(level="time")["y"].transform("mean")
        S["x"] = S["x"] - S.groupby(level="time")["x"].transform("mean")
    b = float((S["x"] * S["y"]).sum() / (S["x"] ** 2).sum())
    S["s"] = S["x"] * (S["y"] - b * S["x"])
    st = S.groupby(level="time")["s"].sum().sort_index().values
    L = max(24, 2 * k)
    v = np.sum(st ** 2)
    for lag in range(1, L + 1):
        v += 2 * (1 - lag / (L + 1)) * np.sum(st[lag:] * st[:-lag])
    se = float(np.sqrt(v) / (S["x"] ** 2).sum())
    return {"b": b, "se": se, "t": b / se, "n_obs": int(len(S)), "n_hours": int(len(st))}


def pooled_decomp(series: dict[str, pd.DataFrame], k: int) -> dict:
    """Split each coin's standardised imbalance into the cross-sectional mean of the hour (market flow)
    and the deviation from it (coin-specific flow); pooled regression of the k-hour return on both
    with coin fixed effects and Driscoll-Kraay standard errors."""
    S = pd.concat({s: d[["x", f"y{k}"]] for s, d in series.items()}, names=["coin", "time"]).dropna()
    S = S.rename(columns={f"y{k}": "y"})
    S["m"] = S.groupby(level="time")["x"].transform("mean")
    S["i"] = S["x"] - S["m"]
    for c in ["y", "m", "i"]:
        S[c] = S[c] - S.groupby(level="coin")[c].transform("mean")
    X = S[["m", "i"]].to_numpy()
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ S["y"].to_numpy()
    e = S["y"].to_numpy() - X @ b
    sc = pd.DataFrame(X * e[:, None], index=S.index).groupby(level="time").sum().sort_index().to_numpy()
    L = max(24, 2 * k)
    V = sc.T @ sc
    for lag in range(1, L + 1):
        G = sc[lag:].T @ sc[:-lag]
        V += (1 - lag / (L + 1)) * (G + G.T)
    cov = XtX_inv @ V @ XtX_inv
    se = np.sqrt(np.diag(cov))
    # report per one SD of each component
    sd_m, sd_i = S["m"].std(), S["i"].std()
    return {"b_market": b[0] * sd_m, "t_market": b[0] / se[0], "b_coin": b[1] * sd_i, "t_coin": b[1] / se[1]}


def pooled_response(rows: pd.DataFrame, series: dict[str, pd.DataFrame]) -> dict:
    from scipy.stats import binomtest
    out = {"n_coins": int(len(rows))}
    for k in (1, 2, 4):
        for tag, sub in [("all", series), ("exbtc", {s: d for s, d in series.items() if s != "BTCUSDT"})]:
            for fe in (False, True):
                r = pooled_dk(sub, k, time_fe=fe)
                key = f"{tag}_{'tfe' if fe else 'cfe'}_k{k}"
                out[f"{key}_b"], out[f"{key}_se"], out[f"{key}_t"] = r["b"], r["se"], r["t"]
        dec = pooled_decomp(series, k)
        for kk, v in dec.items():
            out[f"decomp_{kk}_k{k}"] = v
        b = rows[f"fr_taker_b{k}"]
        out[f"n_negative_{k}"] = int((b < 0).sum())
        out[f"n_sig_negative_{k}"] = int((rows[f"fr_taker_t{k}"] < -1.96).sum())
        out[f"n_sig_positive_{k}"] = int((rows[f"fr_taker_t{k}"] > 1.96).sum())
        out[f"sign_test_p_{k}"] = float(binomtest(int((b < 0).sum()), len(b), 0.5).pvalue)
    out["n_sp_taker_sig"] = int((rows["sp_taker_cwp_2"] < 0.05).sum())
    out["n_sp_bvc_sig"] = int((rows["sp_bvc_cwp_2"] < 0.05).sum())
    return out


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    syms = sorted(p.name.replace("_1m.csv.gz", "") for p in PANEL.glob("*_1m.csv.gz"))
    if len(sys.argv) > 1:  # optional explicit list
        syms = [s for s in syms if s in sys.argv[1:]]
    syms = ["BTCUSDT"] + [s for s in syms if s != "BTCUSDT"]
    print(f"[panel] {len(syms)} coins")
    res = {s: one_asset(s, cfg) for s in syms}
    rows = pd.DataFrame([r for r, _ in res.values()])
    rows.to_csv(OUT / "table_panel.csv", index=False)
    from msr import plots
    (OUT / "figures").mkdir(exist_ok=True)
    plots.fig_panel_ceiling(rows, OUT / "figures" / "fig_panel_ceiling.png")
    pooled = pooled_response(rows, {s: ser for s, (_, ser) in res.items()})
    pd.Series(pooled).to_csv(OUT / "pooled_response.csv", header=["value"])
    print(rows[["symbol", "corr_bvc_1h", "ceil_is_1h", "ceil_is_1m", "corr_gk_rv", "fr_taker_b2", "fr_taker_t2",
                "sp_taker_cwp_2"]].round(3).to_string())
    print(pooled)


if __name__ == "__main__":
    main()
