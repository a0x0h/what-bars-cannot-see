"""Build the formatted LaTeX tables used in paper/main.tex from results/<ASSET>/*.csv.

    python paper/make_tables.py

Writes paper/tables/*.tex. Every number in the paper's tables comes from here;
numbers quoted in the text are taken from the same CSV files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)

NAMES = {"mean": "Historical mean", "enet_TI": "EN: TI", "enet_MSproxy": r"EN: MS$_{\text{proxy}}$",
         "enet_MS": "EN: MS", "enet_ALL": "EN: ALL", "enet_FLOWproxy": r"EN: Flow$_{\text{BVC}}$",
         "enet_FLOWtrue": r"EN: Flow$_{\text{taker}}$", "gbm_MS": "GBM: MS", "gbm_ALL": "GBM: ALL",
         "buy_and_hold": "Buy and hold"}


def _res(asset):
    for cand in (asset, f"{asset}USDT_1H", f"{asset}USDT"):
        if (RES / cand).exists():
            return RES / cand
    raise FileNotFoundError(asset)


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def fnum(x, d=3):
    """Format a number for LaTeX; negative values get a true minus sign that works in text and math."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{d}f}"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def write(name, body):
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def descriptive():
    from msr.data import load_ohlcv
    rows = []
    for sym in ["BTCUSDT", "ETHUSDT"]:
        p = ROOT / "data" / "raw" / f"{sym}_1h.csv"
        if not p.exists():
            continue
        df = load_ohlcv(p, verbose=False)
        r4 = np.log(df["close"]).diff(4)
        r1 = np.log(df["close"]).diff()
        imb = (2 * df["taker_buy_volume"] - df["volume"]) / df["volume"]
        dv = df["close"] * df["volume"] / 1e6
        ac = imb.autocorr(1)
        rows.append([sym.replace("USDT", "/USDT"), f"{df.index[0]:%Y-%m-%d}", f"{df.index[-1]:%Y-%m-%d}",
                     f"{df['close'].notna().sum():,}", f"{df.attrs['n_missing_hours']:,}",
                     fnum(1e4 * r1.std(), 1), fnum(1e4 * r4.std(), 1), fnum(r1.kurt(), 1),
                     fnum(dv.median(), 1), f"{df['trades'].median():,.0f}",
                     fnum(imb.mean(), 3), fnum(imb.std(), 3), fnum(ac, 2),
                     fnum(imb.corr(r1), 2)])
    hdr = ("Asset & Start & End & Hours & Missing & $\\sigma(r_{1h})$ & $\\sigma(r_{4h})$ & Kurt$(r_{1h})$ & "
           "Med.\\ \\$vol (M) & Med.\\ trades & Mean imb. & SD imb. & AC(1) imb. & Corr(imb., $r_{1h}$)")
    body = "\\begin{tabular}{l" + "r" * 13 + "}\n\\toprule\n" + hdr + "\\\\\n\\midrule\n"
    body += "\n".join(" & ".join(r) + "\\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write("tab_descriptive", body)


MACROS: dict[str, str] = {}
BS = "\\"
NL = "\\\\\n"  # LaTeX row end + newline


def macro(name, value):
    MACROS[name] = value


def write_macros():
    bad = [k for k in MACROS if not k.isalpha()]
    assert not bad, f"LaTeX macro names must be letters only: {bad}"
    body = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items()))
    (OUT / "numbers.tex").write_text(body + "\n", encoding="utf-8")
    print(f"wrote numbers.tex ({len(MACROS)} macros)")


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


def tab(cols, header, lines):
    return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header}{NL}\\midrule\n" + "".join(lines)
            + "\\bottomrule\n\\end{tabular}\n")


def validation_tables(asset="BTC", main=True):
    d = _res(asset)
    P = asset
    pv = pd.read_csv(d / "table_proxy_validation.csv")
    cv = pd.read_csv(d / "table_classifier_variants.csv")
    ic = pd.read_csv(d / "table_information_ceiling.csv").set_index("bars")
    sd = pd.read_csv(d / "table_spread_diagnostics.csv").set_index("estimator")
    rg = pd.read_csv(d / "table_validation_by_regime.csv")

    def pvrow(q, agg):
        return pv[(pv["quantity"].str.startswith(q)) & (pv["aggregation"] == agg)].iloc[0]

    b1, b4, b24 = (pvrow("order-flow imbalance: BVC(1h", a) for a in ("1h", "4h", "24h"))
    m1h = pvrow("order-flow imbalance: BVC(1m", "1h")
    lam = pvrow("price impact", "48h")
    gk = pvrow("volatility", "24h")
    macro(f"{P}bvcCorr", fnum(b1.pearson, 2)); macro(f"{P}bvcSign", pct(b1.sign_agreement))
    macro(f"{P}bvcCorrDay", fnum(b24.pearson, 2)); macro(f"{P}bvcOneMinCorr", fnum(m1h.pearson, 2))
    macro(f"{P}lambdaCorr", fnum(lam.pearson, 2)); macro(f"{P}gkCorr", fnum(gk.pearson, 2))
    macro(f"{P}ceilH", fnum(ic.loc["1h", "R2_ceiling_any_f(z)"], 2))
    macro(f"{P}bvcResH", fnum(ic.loc["1h", "R2_BVC_rescaled"], 2))
    macro(f"{P}ceilM", fnum(ic.loc["1m", "R2_ceiling_any_f(z)"], 2))
    macro(f"{P}bvcResM", fnum(ic.loc["1m", "R2_BVC_rescaled"], 2))
    macro(f"{P}slopeH", fnum(ic.loc["1h", "BVC_slope"], 2))
    cs = sd.loc["Corwin-Schultz (24h mean)"]
    macro(f"{P}tick", f"{cs.one_tick_bps:.3f}"); macro(f"{P}csMedian", fnum(cs.median_bps, 1))
    macro(f"{P}csVolCorr", fnum(cs["corr_log_with_rv_1m_24"], 2))
    macro(f"{P}arZero", pct(sd.loc["Abdi-Ranaldo (24h)", "share_zero_%"] / 100))
    macro(f"{P}rollZero", pct(sd.loc["Roll (48h)", "share_zero_%"] / 100))
    yr = rg[(rg.split == "year") & (rg.aggregation == "1h")]
    macro(f"{P}bvcYearMin", fnum(yr.pearson.min(), 2)); macro(f"{P}bvcYearMax", fnum(yr.pearson.max(), 2))
    fee = rg[(rg.split == "fee_regime") & (rg.aggregation == "1h")].set_index("group")
    macro(f"{P}bvcZeroFee", fnum(fee.loc["zero-fee promo", "pearson"], 2))
    macro(f"{P}bvcStdFee", fnum(fee.loc["standard fees", "pearson"], 2))

    def panel(t):
        return f"\\multicolumn{{5}}{{l}}{{\\emph{{{t}}}}}{NL}"

    def row(lab, a, b, c, n):
        return f"{lab} & {fnum(a)} & {fnum(b)} & {fnum(c)} & {int(n):,}{NL}"

    tr = cv[(cv.classifier == "Tick rule (1h bars)") & (cv.aggregation == "1h")].iloc[0]
    L = [panel("Panel A. Signed order flow: proxy vs.\\ observed taker imbalance")]
    for lab, r in [("BVC, 1h bars, per hour", b1), ("BVC, 1h bars, 4h sums", b4), ("BVC, 1h bars, 24h sums", b24),
                   ("BVC, 1m bars summed to 1h", m1h), ("Tick rule, 1h bars, per hour", tr)]:
        L.append(row(lab, r.pearson, r.spearman, r.sign_agreement, r.n))
    L.append("\\midrule\n" + panel("Panel B. Price impact and volatility"))
    L.append(row("Kyle $\\lambda$ (48h): BVC vs.\\ observed flow", lam.pearson, lam.spearman, np.nan, lam.n))
    L.append(row("log GK vol (1h bars) vs.\\ log RV (1m), 24h", gk.pearson, gk.spearman, np.nan, gk.n))
    L.append("\\midrule\n" + panel("Panel C. Information ceiling of return-based classifiers (out-of-sample $R^2$)"))
    L.append(f" & BVC raw & BVC rescaled & Any $f(z)$ & $N${NL}")
    for b, lab in [("1h", "1h bars"), ("1m", "1m bars")]:
        if b in ic.index:
            r = ic.loc[b]
            L.append(row(lab, r.R2_BVC_raw, r.R2_BVC_rescaled, r["R2_ceiling_any_f(z)"], r.n_eval))
    L.append("\\midrule\n" + panel(f"Panel D. Spread estimators (one tick $=$ {cs.one_tick_bps:.3f} bps)"))
    L.append(f" & Median (bps) & Mean (bps) & Share $=0$ & Corr.\\ log RV{NL}")
    for est, r in sd.iterrows():
        L.append(f"{est} & {fnum(r.median_bps, 1)} & {fnum(r.mean_bps, 1)} & {fnum(r['share_zero_%'], 1)}\\% & "
                 f"{fnum(r['corr_log_with_rv_1m_24'], 2)}{NL}")
    write(f"tab_validation{'' if main else '_' + asset.lower()}",
          tab("lrrrr", " & Pearson & Spearman & Same sign & $N$", L))

    piv = cv.pivot_table(index="classifier", columns="aggregation", values=["pearson", "sign_agreement"], sort=False)
    L = [f"{k} & {fnum(r[('pearson', '1h')])} & {fnum(r[('pearson', '24h')])} & {fnum(r[('sign_agreement', '1h')])} & "
         f"{fnum(r[('sign_agreement', '24h')])}{NL}" for k, r in piv.iterrows()]
    hdr = (f" & \\multicolumn{{2}}{{c}}{{Pearson}} & \\multicolumn{{2}}{{c}}{{Same sign}}{NL}"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5} Classifier & 1h & 24h & 1h & 24h")
    write(f"tab_variants_{asset.lower()}", tab("lrrrr", hdr, L))

    piv = rg.pivot_table(index=["split", "group"], columns="aggregation", values=["pearson", "sign_agreement", "n"], sort=False)
    L = [f"{sp.replace('_', ' ')} & {g} & {fnum(r[('pearson', '1h')])} & {fnum(r[('pearson', '24h')])} & "
         f"{fnum(r[('sign_agreement', '1h')])} & {fnum(r[('sign_agreement', '24h')])} & {int(r[('n', '1h')]):,}{NL}"
         for (sp, g), r in piv.iterrows()]
    hdr = (f" & & \\multicolumn{{2}}{{c}}{{Pearson}} & \\multicolumn{{2}}{{c}}{{Same sign}} & {NL}"
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6} Split & Group & 1h & 24h & 1h & 24h & Hours")
    write(f"tab_regimes_{asset.lower()}", tab("llrrrrr", hdr, L))


def zero_fee_activity():
    from msr.data import load_ohlcv
    for sym, P in [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")]:
        p = ROOT / "data" / "raw" / f"{sym}_1h.csv"
        if not p.exists():
            continue
        df = load_ohlcv(p, verbose=False)
        t = df.index
        from msr.events import TUSD_OFF, ZERO_FEE_OFF, ZERO_FEE_ON
        windows = {"Pre": (pd.Timestamp("2022-01-01", tz="UTC"), ZERO_FEE_ON), "Zero": (ZERO_FEE_ON, ZERO_FEE_OFF),
                   "Post": (ZERO_FEE_OFF, TUSD_OFF)}
        for lab, (a, b) in windows.items():
            m = (t >= a) & (t < b)
            macro(f"{P}trades{lab}", f"{df.loc[m, 'trades'].median():,.0f}")
            macro(f"{P}dvol{lab}", f"{(df.loc[m, 'close'] * df.loc[m, 'volume']).median() / 1e6:,.0f}")


MODEL_ORDER = ["mean", "enet_TI", "enet_MSproxy", "enet_MS", "enet_ALL", "enet_FLOWproxy", "enet_FLOWtrue",
               "gbm_MS", "gbm_ALL"]


def tstat(t, p):
    return "--" if not np.isfinite(t) else fnum(t, 2) + "$" + stars(p) + "$"


def forecast_tables(asset="BTC", main=True):
    d = _res(asset)
    P = asset
    sfx = "" if main else "_" + asset.lower()
    ft = pd.read_csv(d / "table_forecast_accuracy.csv").set_index("model")
    L = []
    for m in [m for m in MODEL_ORDER if m in ft.index]:
        r = ft.loc[m]
        L.append(f"{NAMES[m]} & {fnum(r['R2oos_vs_zero_%'])} & {fnum(r['R2oos_vs_mean_%'])} & "
                 f"{tstat(r.CW_t_vs_zero, r.CW_p_vs_zero)} & {fnum(r.IC)} & [{fnum(r.IC_lo95)}, {fnum(r.IC_hi95)}] & "
                 f"{fnum(r.hit_rate)} & {fnum(r.hit_t, 2)}{NL}")
    hdr = ("Model & $R^2_{OOS}$ vs.\\ 0 (\\%) & $R^2_{OOS}$ vs.\\ mean (\\%) & CW $t$ vs.\\ 0 & IC & 95\\% CI & "
           "Hit rate & Hit $t$")
    write(f"tab_forecast{sfx}", tab("lrrrrcrr", hdr, L))
    best = ft.drop(index="mean")["R2oos_vs_zero_%"]
    macro(f"{P}bestRtwo", fnum(best.max(), 3)); macro(f"{P}bestRtwoModel", NAMES[best.idxmax()])
    macro(f"{P}nNegRtwo", str(int((best < 0).sum()))); macro(f"{P}nModels", str(len(best)))
    for m in ["enet_TI", "enet_ALL", "enet_MS", "enet_FLOWtrue", "enet_FLOWproxy"]:
        if m in ft.index:
            k = m.replace("enet_", "").replace("_", "")
            macro(f"{P}cwp{k}", fnum(ft.loc[m, "CW_p_vs_zero"], 3))
            macro(f"{P}ic{k}", fnum(ft.loc[m, "IC"], 3))
            macro(f"{P}rtwo{k}", fnum(ft.loc[m, "R2oos_vs_zero_%"], 3))

    inc = pd.read_csv(d / "table_incremental.csv")
    L = []
    for _, r in inc.iterrows():
        L.append(f"{NAMES[r.model]} & {NAMES[r.vs]} & {'yes' if r.nested else 'no'} & {fnum(r['R2oos_rel_%'])} & "
                 f"{tstat(r.CW_t, r.CW_p)} & {fnum(r.CW_p)} & {tstat(r.DM_t, r.DM_p)} & {fnum(r.DM_p)}{NL}")
        k = f"{r.model}{r.vs}".replace("enet_", "").replace("gbm_", "G").replace("_", "")
        macro(f"{P}inc{k}R", fnum(r["R2oos_rel_%"], 3)); macro(f"{P}inc{k}CWp", fnum(r.CW_p, 3))
        macro(f"{P}inc{k}DMp", fnum(r.DM_p, 3))
    hdr = "Model & Benchmark & Nested & $R^2_{OOS}$ rel.\\ (\\%) & CW $t$ & CW $p$ & DM $t$ & DM $p$"
    write(f"tab_incremental{sfx}", tab("llcrrrrr", hdr, L))

    bt = pd.read_csv(d / "table_backtest.csv")
    sd = pd.read_csv(d / "table_strategy_diagnostics.csv").set_index(["model", "rule"])
    L = []
    for rule in ["sign", "selective"]:
        L.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{'Sign rule' if rule == 'sign' else 'Selective rule (top quintile of |forecast|)'}}}}}{NL}")
        for m in [m for m in MODEL_ORDER if m != "mean" and m in set(bt.model)]:
            g = bt[(bt.model == m) & (bt.rule == rule)].set_index("cost_bps")
            s = sd.loc[(m, rule)]
            L.append(f"{NAMES[m]} & {fnum(g.loc[0, 'ann_ret_%'], 1)} & {fnum(g.loc[0, 'sharpe'], 2)} & "
                     f"{fnum(g.loc[2, 'sharpe'], 2)} & {fnum(g.loc[5, 'sharpe'], 2)} & {fnum(g.loc[0, 'turnover_per_day'], 2)} & "
                     f"{fnum(g.loc[0, 'breakeven_bps'], 1)} & {fnum(s.alpha_t, 2)} & {s.pos_years} & {fnum(s.DSR, 2)}{NL}")
    bh = bt[bt.model == "buy_and_hold"].iloc[0]
    L.append(f"\\midrule\nBuy and hold & {fnum(bh['ann_ret_%'], 1)} & {fnum(bh.sharpe, 2)} & & & & & & & {NL}")
    hdr = ("& Ann.\\ ret.\\ (\\%) & \\multicolumn{3}{c}{Sharpe at cost (bps)} & Turnover & Break-even & "
           f"$\\alpha$ $t$ & Years & DSR{NL}\\cmidrule(lr){{3-5}} Model & gross & 0 & 2 & 5 & per day & (bps) & (HAC) & $SR>0$ & ")
    write(f"tab_backtest{sfx}", tab("lrrrrrrrcr", hdr, L))
    bm, br = sd["sharpe"].idxmax()
    b = sd.loc[(bm, br)]
    g = bt[(bt.model == bm) & (bt.rule == br)].set_index("cost_bps")
    macro(f"{P}bestModel", NAMES[bm]); macro(f"{P}bestRule", br)
    macro(f"{P}bestSR", fnum(b.sharpe, 2)); macro(f"{P}bestDSR", fnum(b.DSR, 2)); macro(f"{P}bestPSR", fnum(b.PSR_vs_0, 3))
    macro(f"{P}bestAlphaT", fnum(b.alpha_t, 1)); macro(f"{P}bestBeta", fnum(b.beta, 2))
    macro(f"{P}bestYears", b.pos_years); macro(f"{P}bestSRmin", fnum(b.sharpe_min_year, 1))
    macro(f"{P}bestSRmax", fnum(b.sharpe_max_year, 1)); macro(f"{P}bestBE", fnum(g.loc[0, "breakeven_bps"], 1))
    macro(f"{P}bestSRfive", fnum(g.loc[5, "sharpe"], 2)); macro(f"{P}nTrials", str(int(b.n_trials)))
    macro(f"{P}SRzero", fnum(b.SR0_ann, 2)); macro(f"{P}bhSR", fnum(bh.sharpe, 2))
    macro(f"{P}maxBE", fnum(bt[bt.model != "buy_and_hold"]["breakeven_bps"].max(), 1))
    macro(f"{P}nPosSRfive",str(int((bt[(bt.cost_bps == 5) & (bt.model != 'buy_and_hold')].sharpe > 0).sum())))

    # Null-model share of the elastic net
    c = pd.read_csv(d / "enet_coefficients_by_fold.csv")
    c = c.rename(columns={c.columns[0]: "fold", c.columns[1]: "model"})
    nz = (c.drop(columns=["fold", "model"]).fillna(0).abs() > 0).sum(axis=1)
    share = (nz == 0).groupby(c["model"]).mean()
    for m, v in share.items():
        macro(f"{P}null{m.replace('enet_', '').replace('_', '')}", pct(v))
    macro(f"{P}nFolds", str(c["fold"].nunique()))


def robustness_tables(asset="BTC"):
    d = _res(asset)
    P = asset
    if (d / "table_horizons.csv").exists():
        hz = pd.read_csv(d / "table_horizons.csv")
        ms = [m for m in MODEL_ORDER if m in set(hz.model)]
        L = []
        for h, g in hz.groupby("horizon_h"):
            g = g.set_index("model")
            L.append(f"{h}h & " + " & ".join(f"{fnum(g.loc[m, 'R2oos_%'])} ({fnum(g.loc[m, 'CW_p'], 2)})" for m in ms) + NL)
        write("tab_horizons", tab("l" + "r" * len(ms), "Horizon & " + " & ".join(NAMES[m] for m in ms), L))
        ti = hz[hz.model == "enet_TI"].set_index("horizon_h")
        for h, w in [(1, "one"), (4, "four"), (8, "eight"), (24, "day")]:
            if h in ti.index:
                macro(f"{P}hzTI{w}", fnum(ti.loc[h, "R2oos_%"], 3)); macro(f"{P}hzTI{w}p", fnum(ti.loc[h, "CW_p"], 3))
    if (d / "table_rolling_window.csv").exists():
        rw = pd.read_csv(d / "table_rolling_window.csv").set_index("model")
        ex = pd.read_csv(d / "table_forecast_accuracy.csv").set_index("model")
        L = [f"{NAMES[m]} & {fnum(ex.loc[m, 'R2oos_vs_zero_%'])} & {fnum(rw.loc[m, 'R2oos_vs_zero_%'])} & "
             f"{fnum(rw.loc[m, 'CW_p_vs_zero'], 2)} & {fnum(rw.loc[m, 'IC'])}{NL}" for m in MODEL_ORDER if m in rw.index]
        write("tab_rolling", tab("lrrrr", "Model & Expanding $R^2_{OOS}$ & Rolling $R^2_{OOS}$ & Rolling CW $p$ & Rolling IC", L))
    if (d / "table_drop_group.csv").exists():
        dg = pd.read_csv(d / "table_drop_group.csv")
        base = pd.read_csv(d / "table_forecast_accuracy.csv").set_index("model").loc["enet_ALL"]
        L = [f"none (EN: ALL) & {int(len(pd.read_csv(d / 'enet_coefficients_by_fold.csv').columns) - 2)} & "
             f"{fnum(base['R2oos_vs_zero_%'])} & {fnum(base.IC)}{NL}"]
        L += [f"{r.dropped_group.replace('_', ' ')} & {r.n_features} & {fnum(r['R2oos_%'])} & {fnum(r.IC)}{NL}"
              for _, r in dg.iterrows()]
        write("tab_dropgroup", tab("lrrr", "Group dropped & Features & $R^2_{OOS}$ (\\%) & IC", L))
    sp = pd.read_csv(d / "table_subperiods.csv")
    ms = [m for m in MODEL_ORDER if m in set(sp.model)]
    L = []
    for per, g in sp.groupby("subperiod", sort=False):
        g = g.set_index("model")
        L.append(f"{per.replace(' to ', '--')} & {fnum(g['market_ret_%'].iloc[0], 0)} & "
                 + " & ".join(fnum(g.loc[m, 'R2oos_%']) for m in ms) + NL)
    write("tab_subperiods", tab("lr" + "r" * len(ms), "Sub-period & BTC ret.\\ (\\%) & " + " & ".join(NAMES[m] for m in ms), L))
    cd = pd.read_csv(d / "table_conditional.csv")
    ms = [m for m in ["enet_TI", "enet_MS", "enet_ALL", "enet_FLOWtrue"] if m in set(cd.model)]
    L = []
    for (dim, grp), g in cd.groupby(["dimension", "group"], sort=False):
        g = g.set_index("model")
        L.append(f"{dim.replace('_', ' ')} & {grp} & {int(g['n'].iloc[0]):,} & "
                 + " & ".join(f"{fnum(g.loc[m, 'R2oos_%'])} ({fnum(g.loc[m, 'CW_p'], 2)})" for m in ms) + NL)
    write("tab_conditional", tab("llr" + "r" * len(ms), "Split & Group & Hours & " + " & ".join(NAMES[m] for m in ms), L))


def flow_response_tables(asset="BTC"):
    d = _res(asset)
    P = asset
    if not (d / "table_flow_response.csv").exists():
        return
    fr = pd.read_csv(d / "table_flow_response.csv")
    ks = [1, 2, 4, 8, 12, 24]
    L = []
    for spec, reg, lab in [("taker", "taker", "Observed taker imbalance"), ("bvc", "bvc", "BVC imbalance"),
                           ("joint", "taker", "Observed imbalance, given $z_t$"), ("joint", "z", "Bar return $z_t$, given imbalance")]:
        s = fr[(fr.spec == spec) & (fr.regressor == reg)].set_index("k")
        L.append(lab + " & " + " & ".join(f"{fnum(s.loc[k, 'beta_bps'], 2)}" for k in ks) + NL)
        L.append(" & " + " & ".join(f"({fnum(s.loc[k, 't'], 2)})" for k in ks) + NL)
    write(f"tab_flow_response{'' if asset == 'BTC' else '_' + asset.lower()}",
          tab("l" + "r" * len(ks), "Regressor (1 SD) & " + " & ".join(f"$k={k}$h" for k in ks), L))
    for spec, reg, nm in [("taker", "taker", "Tak"), ("bvc", "bvc", "Bvc"), ("joint", "taker", "Joint"), ("joint", "z", "Z")]:
        s = fr[(fr.spec == spec) & (fr.regressor == reg)].set_index("k")
        for k, w in [(1, "one"), (2, "two"), (4, "four"), (8, "eight"), (24, "day")]:
            macro(f"{P}fr{nm}B{w}", fnum(s.loc[k, "beta_bps"], 1))
            macro(f"{P}fr{nm}T{w}", fnum(s.loc[k, "t"], 1))
            macro(f"{P}fr{nm}A{w}", fnum(abs(s.loc[k, "beta_bps"]), 1))
    s = fr[(fr.spec == "taker") & (fr.regressor == "taker")].set_index("k")
    macro(f"{P}frRtwoFour", fnum(s.loc[4, "r2_%"], 3))


def response_split_table():
    L = []
    for asset in ["BTC", "ETH"]:
        try:
            f = _res(asset) / "table_flow_response_splits.csv"
        except FileNotFoundError:
            continue
        if not f.exists():
            continue
        fr = pd.read_csv(f)
        L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{asset}/USDT}}}}{NL}")
        for samp, g in fr.groupby("sample", sort=False):
            cells = []
            for spec in ["taker", "joint"]:
                for k in (2, 4):
                    r = g[(g.spec == spec) & (g.regressor == "taker") & (g.k == k)].iloc[0]
                    cells.append(f"{fnum(r.beta_bps, 2)} ({fnum(r.t, 1)})")
            n = int(g["n"].iloc[0])
            L.append(f"{samp} & {n:,} & " + " & ".join(cells) + NL)
    hdr = (f" & & \\multicolumn{{2}}{{c}}{{Observed imbalance}} & \\multicolumn{{2}}{{c}}{{Observed imbalance $\\mid z_t$}}{NL}"
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6} Sample & Hours & $k=2$ & $k=4$ & $k=2$ & $k=4$")
    write("tab_response_splits", tab("lrrrrr", hdr, L))


HW = {1: "one", 2: "two", 4: "four"}
SPK = {"Observed taker imbalance": "tak", "BVC imbalance (1h bars)": "bvch", "BVC imbalance (1m bars)": "bvcm"}


def single_predictor_table():
    L = []
    for asset in ["BTC", "ETH"]:
        f = _res(asset) / "table_single_predictor_extra.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{asset}/USDT}}}}{NL}")
        for _, r in t.iterrows():
            L.append(f"{r.predictor} & {int(r.horizon_h)}h & {fnum(r['R2oos_%'], 3)} & {tstat(r.CW_t, r.CW_p)} & "
                     f"{fnum(r.CW_p, 4)} & {fnum(r.IC, 3)} & {fnum(r['R2oos_first_half_%'], 3)} ({fnum(r.CW_p_first_half, 3)}) & "
                     f"{fnum(r['R2oos_second_half_%'], 3)} ({fnum(r.CW_p_second_half, 3)}){NL}")
            w, k = HW[int(r.horizon_h)], SPK[r.predictor]
            if k == "tak":
                macro(f"{asset}spR{w}", fnum(r["R2oos_%"], 3)); macro(f"{asset}spP{w}", fnum(r.CW_p, 4))
                macro(f"{asset}spIC{w}", fnum(r.IC, 3))
                macro(f"{asset}spRA{w}", fnum(r["R2oos_first_half_%"], 3)); macro(f"{asset}spPA{w}", fnum(r.CW_p_first_half, 3))
                macro(f"{asset}spRB{w}", fnum(r["R2oos_second_half_%"], 3)); macro(f"{asset}spPB{w}", fnum(r.CW_p_second_half, 4))
            else:
                macro(f"{asset}sp{k}R{w}", fnum(r["R2oos_%"], 3)); macro(f"{asset}sp{k}P{w}", fnum(r.CW_p, 3))
    hdr = (f" & & & & & & \\multicolumn{{2}}{{c}}{{$R^2_{{OOS}}$ (CW $p$) by half}}{NL}\\cmidrule(lr){{7-8}}"
           "Predictor & Horizon & $R^2_{OOS}$ (\\%) & CW $t$ & CW $p$ & IC & first & second")
    write("tab_single", tab("lrrrrrrr", hdr, L))


def single_trading_table():
    rows = []
    for asset in ["BTC", "ETH"]:
        f = _res(asset) / "table_single_predictor_trading.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        for rule, g in t.groupby("rule", sort=False):
            g = g.set_index("cost_bps")
            rows.append(f"{asset}/USDT & {rule} & {fnum(g.loc[0, 'ann_ret_%'], 1)} & {fnum(g.loc[0, 'sharpe'], 2)} & "
                        f"{fnum(g.loc[1, 'sharpe'], 2)} & {fnum(g.loc[2, 'sharpe'], 2)} & "
                        f"{fnum(g.loc[0, 'turnover_per_day'], 1)} & {fnum(g.loc[0, 'breakeven_bps'], 2)}{NL}")
            macro(f"{asset}stBE{rule}", fnum(g.loc[0, "breakeven_bps"], 1))
            macro(f"{asset}stSR{rule}", fnum(g.loc[0, "sharpe"], 2))
    hdr = (f" & & Ann.\\ ret.\\ (\\%) & \\multicolumn{{3}}{{c}}{{Sharpe at cost (bps)}} & Turnover & Break-even{NL}"
           "\\cmidrule(lr){4-6} Asset & Rule & gross & 0 & 1 & 2 & per day & (bps)")
    write("tab_single_trading", tab("llrrrrrr", hdr, rows))


def response_controls_table():
    L = []
    for asset in ["BTC", "ETH"]:
        f = _res(asset) / "table_response_controls.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        L.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{asset}/USDT}}}}{NL}")
        for (spec, reg), g in t.groupby(["spec", "regressor"], sort=False):
            g = g.set_index("k")
            lab = spec + (": 1m BVC" if reg == "bvc_1m" and "vs" in spec else ": taker" if "vs" in spec else "")
            L.append(f"{lab} & " + " & ".join(f"{fnum(g.loc[k, 'beta_bps'], 2)} ({fnum(g.loc[k, 't'], 1)})" for k in (1, 2, 4)) + NL)
            if asset == "BTC":
                key = {"(4) + volume, volatility, lagged taker": "ctrl", "(5) 1m BVC alone": "bvcm"}.get(spec)
                if key:
                    macro(f"BTCrc{key}B", fnum(g.loc[2, "beta_bps"], 1)); macro(f"BTCrc{key}T", fnum(g.loc[2, "t"], 1))
                if spec.startswith("(6)"):
                    k2 = "vsTak" if reg == "taker" else "vsBvcm"
                    macro(f"BTCrc{k2}B", fnum(g.loc[2, "beta_bps"], 1)); macro(f"BTCrc{k2}T", fnum(g.loc[2, "t"], 1))
    write("tab_response_controls", tab("lrrr", "Specification (coefficient of interest) & $k=1$h & $k=2$h & $k=4$h", L))


def ceiling_tables():
    L = []
    for asset in ["BTC", "ETH"]:
        f = _res(asset) / "table_ceiling_in_sample.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        p1 = t[t.bars == "1h"].set_index("period")
        pm = t[t.bars == "1m"].set_index("period")
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{asset}/USDT}}}}{NL}")
        for per in p1.index:
            r, q = p1.loc[per], pm.loc[per] if per in pm.index else None
            L.append(f"{per} & {int(r.n):,} & {fnum(r.R2_ceiling_in_sample)} & {fnum(r.corr2_BVC)} & "
                     f"{fnum(r.slope_imb_on_BVC)} & {fnum(r.sd_imb)} & {fnum(q.R2_ceiling_in_sample) if q is not None else '--'}{NL}")
        yrs = p1[p1.index.str.isnumeric()]
        macro(f"{asset}ceilIS", fnum(p1.loc["full sample", "R2_ceiling_in_sample"], 2))
        macro(f"{asset}ceilISmin", fnum(yrs.R2_ceiling_in_sample.min(), 2))
        macro(f"{asset}ceilISmax", fnum(yrs.R2_ceiling_in_sample.max(), 2))
        macro(f"{asset}ceilISm", fnum(pm.loc["full sample", "R2_ceiling_in_sample"], 2))
        macro(f"{asset}corrsqBVC", fnum(p1.loc["full sample", "corr2_BVC"], 2))
        macro(f"{asset}slopeFirst", fnum(yrs.slope_imb_on_BVC.iloc[0], 3))
        macro(f"{asset}slopeLast", fnum(yrs.slope_imb_on_BVC.iloc[-1], 3))
        macro(f"{asset}sdImbZero", fnum(p1.loc["zero-fee window", "sd_imb"], 3))
        macro(f"{asset}sdImbStd", fnum(p1.loc["standard fees", "sd_imb"], 3))
        macro(f"{asset}slopeZero", fnum(p1.loc["zero-fee window", "slope_imb_on_BVC"], 3))
        macro(f"{asset}slopeStd", fnum(p1.loc["standard fees", "slope_imb_on_BVC"], 3))
        s = pd.read_csv(_res(asset) / "table_ceiling_sensitivity.csv")
        s1 = s[(s.bars == "1h") & (s.bins == 50)].set_index("split")
        macro(f"{asset}ceilRev", fnum(s1.loc["fit 2nd, eval 1st", "R2_ceiling"], 2))
    hdr = ("Period & Hours & Ceiling & corr$^2$(BVC) & Slope & SD imb. & Ceiling (1m)")
    write("tab_ceiling_periods", tab("lrrrrrr", hdr, L))


# ----------------------------------------------------------------------------------------------
# Zero-fee natural experiment (results/FEE) and the 20-coin panel (results/PANEL)
# ----------------------------------------------------------------------------------------------
FEE_KEYS = {"trades": "Trades", "dvol": "Dvol", "size": "Size", "net1m": "Net", "two_sided": "Two",
            "ts_share": "Share", "abs_imb_h": "Imb", "rv": "Rv", "amihud": "Amihud", "lam_bvc": "LamBvc",
            "lam_true": "LamTrue", "r2_true": "Rsq", "corr_bvc": "Corr", "vpin_true": "VpinTrue",
            "vpin_bvc": "VpinBvc", "ac1": "Ac"}
FEE_ROWS = [
    ("Panel A. Trading activity", [("trades", "Trades per day (millions)", lambda v: fnum(v / 1e6, 2)),
                                   ("dvol", "Dollar volume per day (\\$bn)", lambda v: fnum(v / 1e9, 2)),
                                   ("size", "Average trade size (\\$)", lambda v: f"{v:,.0f}")]),
    ("Panel B. Direction of the volume (one-minute taker flow)", [
        ("net1m", "Directional volume $\\sum_m|B_m-S_m|$ (\\$bn)", lambda v: fnum(v / 1e9, 2)),
        ("two_sided", "Two-sided volume (\\$bn)", lambda v: fnum(v / 1e9, 2)),
        ("ts_share", "Two-sided share of volume", lambda v: fnum(v, 2)),
        ("abs_imb_h", "Mean $|$hourly taker imbalance$|$", lambda v: fnum(v, 3))]),
    ("Panel C. Liquidity and flow measures computed from bars", [
        ("amihud", "Amihud ratio, hourly bars ($|r|$ per \\$bn)", lambda v: fnum(v, 3)),
        ("lam_bvc", "Kyle's $\\lambda$, BVC-signed flow (bps per \\$M)", lambda v: fnum(v, 2)),
        ("vpin_bvc", "VPIN, BVC", lambda v: fnum(v, 2))]),
    ("Panel D. The same quantities from observed taker flow", [
        ("lam_true", "Kyle's $\\lambda$, observed flow (bps per \\$M)", lambda v: fnum(v, 2)),
        ("vpin_true", "VPIN, observed flow", lambda v: fnum(v, 3)),
        ("r2_true", "$R^2$ of 1m returns on observed flow", lambda v: fnum(v, 2)),
        ("corr_bvc", "corr(BVC, observed imbalance), 1m", lambda v: fnum(v, 2))]),
    ("Panel E. Prices", [("rv", "Realised volatility (\\% per day)", lambda v: fnum(100 * np.sqrt(v), 2)),
                         ("ac1", "First-order autocorrelation of 1m returns", lambda v: fnum(v, 3))]),
]


def _p(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else fnum(p, 3))


def fee_tables():
    d = RES / "FEE"
    if not (d / "table_did.csv").exists():
        print("skip fee_tables (run run_fee.py)")
        return
    did = pd.read_csv(d / "table_did.csv").set_index("outcome")
    eth = pd.read_csv(d / "table_did_eth.csv").set_index("outcome")
    d60 = pd.read_csv(d / "table_did_60d.csv").set_index("outcome")
    d180 = pd.read_csv(d / "table_did_180d.csv").set_index("outcome")
    L = []
    for title, rows in FEE_ROWS:
        L.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{title}}}}}{NL}")
        for k, lab, f in rows:
            r = did.loc[k]
            L.append(f"{lab} & {f(r.btc_pre)} & {f(r.btc_zero)} & {f(r.btc_post)} & "
                     f"{fnum(r.on, 2)} ({fnum(r.on_t, 1)}) & {_p(r.p_space_on)} & {_p(r.p_time_on)} & "
                     f"{fnum(r.off, 2)} ({fnum(r.off_t, 1)}) & {_p(r.p_space_off)} & {_p(r.p_time_off)}{NL}")
    hdr = (f" & \\multicolumn{{3}}{{c}}{{BTC/USDT, daily median}} & \\multicolumn{{3}}{{c}}{{Fees removed}} & "
           f"\\multicolumn{{3}}{{c}}{{Fees restored}}{NL}"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}"
           "Measure & Before & Zero fee & After & $\\Delta$ ($t$) & $p_{\\text{space}}$ & $p_{\\text{time}}$ & "
           "$\\Delta$ ($t$) & $p_{\\text{space}}$ & $p_{\\text{time}}$")
    write("tab_fee_did", tab("lrrrrrrrrr", hdr, L))

    # robustness: ETH as the only control, 60- and 180-day windows, full windows with and without trend
    L = []
    for ev, title in [("on", "Panel A. Fees removed (8 July 2022)"), ("off", "Panel B. Fees restored (22 March 2023)")]:
        L.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}{NL}")
        for _, rows in FEE_ROWS:
            for k, lab, _ in rows:
                cells = [f"{fnum(t.loc[k, ev], 2)} ({fnum(t.loc[k, ev + '_t'], 1)})" for t in (did, eth, d60, d180)]
                r = did.loc[k]
                cells += [f"{fnum(r[ev + '_full'], 2)} ({fnum(r[ev + '_full_t'], 1)})",
                          f"{fnum(r[ev + '_trend'], 2)} ({fnum(r[ev + '_trend_t'], 1)})"]
                L.append(f"{lab} & " + " & ".join(cells) + NL)
        if ev == "on":
            L.append("\\midrule\n")
    hdr = (f" & \\multicolumn{{4}}{{c}}{{Event windows}} & \\multicolumn{{2}}{{c}}{{Full windows}}{NL}"
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}"
           f"Measure & Main (90 days) & ETH only & 60 days & 180 days & Means & With trend")
    write("tab_fee_robust", tab("lrrrrrr", hdr, L))

    for k, K in FEE_KEYS.items():
        if k not in did.index:
            continue
        r = did.loc[k]
        for ev in ("on", "off"):
            E = "On" if ev == "on" else "Off"
            macro(f"fee{E}{K}", fnum(r[ev], 2))
            macro(f"fee{E}{K}A", fnum(abs(r[ev]), 2))
            macro(f"fee{E}{K}T", fnum(r[f"{ev}_t"], 1))
            macro(f"fee{E}{K}Ps", _p(r[f"p_space_{ev}"]))
            macro(f"fee{E}{K}Pt", _p(r[f"p_time_{ev}"]))
            if r["transform"] == "log":
                macro(f"fee{E}{K}Pct", fnum(100 * (np.exp(r[ev]) - 1), 0))
                macro(f"fee{E}{K}PctA", fnum(abs(100 * (np.exp(r[ev]) - 1)), 0))
                macro(f"fee{E}{K}X", fnum(np.exp(r[ev]), 1))
            macro(f"fee{E}{K}Eth", fnum(eth.loc[k, ev], 2))
            macro(f"fee{E}{K}EthT", fnum(eth.loc[k, f"{ev}_t"], 1))
        macro(f"fee{K}Pre", fnum(r.btc_pre, 3)); macro(f"fee{K}Zero", fnum(r.btc_zero, 3)); macro(f"fee{K}Post", fnum(r.btc_post, 3))
    n_ctrl = len(list((d / "daily").glob("*.csv*"))) - 1
    macro("feeNcontrols", str(n_ctrl))
    macro("feeNtime", str(int(did["n_time_placebos"].iloc[0])))
    macro("feeMinPspace", fnum(1 / (n_ctrl + 1), 2))

    dec = pd.read_csv(d / "table_decomposition.csv", index_col=0)
    macro("feeExcessTwoShare", pct(dec.loc["two_sided", "share_of_excess_dollar_volume"]))
    macro("feeExcessNetShare", pct(dec.loc["net1m", "share_of_excess_dollar_volume"]))
    macro("feeExcessDvol", fnum(dec.loc["dvol", "excess"] / 1e9, 2))
    macro("feeExcessTrades", fnum(dec.loc["trades", "excess"] / 1e6, 1))
    macro("feeCtrlGrowth", fnum(dec.loc["dvol", "control_growth"], 2))

    import json
    j = json.loads((d / "switch_jumps.json").read_text())
    for ev, E in [("on", "On"), ("off", "Off")]:
        macro(f"feeJump{E}Trades", fnum(j[ev]["trades"], 2))
        macro(f"feeJump{E}TradesA", fnum(abs(j[ev]["trades"]), 2))
        macro(f"feeJump{E}TradesX", fnum(np.exp(j[ev]["trades"]), 2 if j[ev]["trades"] < 0 else 1))
        macro(f"feeJump{E}Dvol", fnum(j[ev]["dvol"], 2))
        macro(f"feeJump{E}Imb", fnum(j[ev]["abs_imb"], 2))
    macro("feeJumpPlaceboMax", fnum(j["placebo_trades_abs_max"], 2))
    macro("feeJumpPlaceboPnn", fnum(j["placebo_trades_abs_p99"], 2))
    macro("feeJumpPlaceboN", f"{j['n_placebo_hours_trades']:,}")

    pdl = pd.read_csv(d / "btc_pairs_daily.csv", parse_dates=["day"])
    f0 = pdl[(pdl.pair == "BTC/FDUSD") & (pdl.dvol > 0)]["day"].min()
    if pd.notna(f0):
        macro("pairFdusdStart", f"{f0:%B %Y}")
    pr = pd.read_csv(d / "table_btc_pairs.csv")
    wins = list(dict.fromkeys(pr["window"]))
    dates = ["Jan--Jul 2022", "Jul 2022--Mar 2023", "Mar--Sep 2023", "Sep--Dec 2023"]
    regimes = ["fees on all", "13 pairs free", "only TUSD free", "FDUSD free"]
    order = [p for p in ["BTC/USDT", "BTC/BUSD", "BTC/TUSD", "BTC/FDUSD", "BTC/USDC"] if p in set(pr["pair"])]

    def musd(v):
        return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"

    L = []
    for p in order:
        g = pr[pr.pair == p].set_index("window")
        cells = []
        for w in wins:
            r = g.loc[w]
            cells.append("--" if not r.days else musd(r.median_dvol_musd))
            cells.append("--" if not r.days else fnum(r.mean_vw_abs_imb, 3))
        L.append(f"{p} & " + " & ".join(cells) + NL)
    hdr = (" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{w}}}" for w in dates[:len(wins)]) + NL
           + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\emph{{{w}}}}}" for w in regimes[:len(wins)]) + NL
           + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(wins)))
           + "Pair & " + " & ".join(["\\$M/day & $|$imb.$|$"] * len(wins)))
    write("tab_btc_pairs", tab("l" + "rr" * len(wins), hdr, L))
    g = pr.set_index(["pair", "window"])
    for p, P in [("BTC/USDT", "Usdt"), ("BTC/TUSD", "Tusd"), ("BTC/BUSD", "Busd"), ("BTC/FDUSD", "Fdusd")]:
        for i, w in enumerate(wins):
            if (p, w) in g.index and g.loc[(p, w), "days"]:
                macro(f"pair{P}Vol{'ABCD'[i]}", musd(g.loc[(p, w), "median_dvol_musd"]))
                macro(f"pair{P}Imb{'ABCD'[i]}", fnum(g.loc[(p, w), "mean_vw_abs_imb"], 3))

    f = d / "table_trade_forensics.csv"
    if f.exists():
        t = pd.read_csv(f)
        a = t.groupby(["regime", "symbol"], sort=False).mean(numeric_only=True)
        L = []
        for (reg, sym), r in a.iterrows():
            L.append(f"{reg} & {sym.replace('USDT', '/USDT')} & {r.n_agg / 1e6:.2f} & {r.median_trade_usd:,.0f} & "
                     f"{pct(r.pingpong_trades, 1)} & {pct(r.pingpong_volume, 1)} & {pct(r.two_sided_1s, 0)} & "
                     f"{pct(r.two_sided_1m, 0)}{NL}")
        hdr = ("Days & Pair & Trades (M/day) & Median trade (\\$) & Round trips: trades & volume & "
               "Two-sided, 1s & Two-sided, 1m")
        write("tab_trade_forensics", tab("llrrrrrr", hdr, L))
        for reg, R in [("fees (before removal)", "A"), ("zero fees (after removal)", "B"),
                       ("zero fees (before restoration)", "C"), ("fees (after restoration)", "D")]:
            for sym, S in [("BTCUSDT", "Btc"), ("ETHUSDT", "Eth")]:
                if (reg, sym) in a.index:
                    r = a.loc[(reg, sym)]
                    macro(f"tf{S}Ping{R}", pct(r.pingpong_trades, 1))
                    macro(f"tf{S}PingVol{R}", pct(r.pingpong_volume, 1))
                    macro(f"tf{S}TwoSec{R}", pct(r.two_sided_1s, 0))
                    macro(f"tf{S}Size{R}", f"{r.median_trade_usd:,.0f}")
                    macro(f"tf{S}N{R}", f"{r.n_agg / 1e6:.1f}")
        e = t[t.symbol == "ETHUSDT"]
        macro("tfEthTwoSecMin", pct(e.two_sided_1s.min(), 0)); macro("tfEthTwoSecMax", pct(e.two_sided_1s.max(), 0))
        b = t[t.symbol == "BTCUSDT"]
        macro("tfBtcPingVolMax", pct(b.pingpong_volume.max(), 1))


def fig_fee_copy():
    """Copy the fee-experiment figures next to the paper."""
    import shutil
    for src in (RES / "FEE" / "figures", RES / "PANEL" / "figures"):
        if src.exists():
            (ROOT / "paper" / "figures").mkdir(exist_ok=True)
            for p in src.glob("*.png"):
                shutil.copy(p, ROOT / "paper" / "figures" / p.name)
                print("copied", p.name)


def panel_tables():
    d = RES / "PANEL"
    if not (d / "table_panel.csv").exists():
        print("skip panel_tables (run run_panel.py)")
        return
    t = pd.read_csv(d / "table_panel.csv").sort_values("median_dvol_musd", ascending=False)
    po = pd.read_csv(d / "pooled_response.csv", index_col=0)["value"]
    L = []
    for _, r in t.iterrows():
        L.append(f"{r.symbol.replace('USDT', '')} & {r.median_dvol_musd:,.1f} & {fnum(r.corr_bvc_1h, 2)} & "
                 f"{fnum(r.corr_bvc1m_1h, 2)} & {fnum(r.ceil_is_1h, 2)} & {fnum(r.ceil_is_1m, 2)} & "
                 f"{fnum(r.corr_gk_rv, 2)} & {fnum(r.cs_median_bps / r.tick_bps, 0)} & "
                 f"{fnum(r.fr_taker_b2, 2)} ({fnum(r.fr_taker_t2, 1)}) & {fnum(r.fr_joint_b2, 2)} ({fnum(r.fr_joint_t2, 1)}) & "
                 f"{_p(r.sp_taker_cwp_2)}{NL}")
    L.append("\\midrule\n")
    for lab, fn in [("Median", "median"), ("Min", "min"), ("Max", "max")]:
        s = getattr(t.select_dtypes("number"), fn)()
        L.append(f"{lab} & {s.median_dvol_musd:,.1f} & {fnum(s.corr_bvc_1h, 2)} & {fnum(s.corr_bvc1m_1h, 2)} & "
                 f"{fnum(s.ceil_is_1h, 2)} & {fnum(s.ceil_is_1m, 2)} & {fnum(s.corr_gk_rv, 2)} & "
                 f"{fnum(getattr(t.cs_median_bps / t.tick_bps, fn)(), 0)} & {fnum(s.fr_taker_b2, 2)} & {fnum(s.fr_joint_b2, 2)} & {NL}")
    hdr = ("Coin & \\$vol & \\multicolumn{2}{c}{corr(BVC, observed)} & \\multicolumn{2}{c}{Ceiling} & corr(GK, & CS / & "
           f"\\multicolumn{{2}}{{c}}{{2h response to 1 SD flow, bps ($t$)}} & SP{NL}"
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){9-10}"
           " & \\$M/h & 1h bars & 1m bars & 1h & 1m & RV) & tick & observed & given $z_t$ & CW $p$")
    write("tab_panel", tab("lrrrrrrrrrr", hdr, L))
    n = len(t)
    macro("panelN", str(n))
    macro("panelVolMin", f"{t.median_dvol_musd.min():,.1f}")
    macro("panelVolMax", f"{t.median_dvol_musd.max():,.0f}")
    for col, K, dgt in [("corr_bvc_1h", "Corr", 2), ("corr_bvc1m_1h", "CorrM", 2), ("ceil_is_1h", "Ceil", 2),
                        ("ceil_is_1m", "CeilM", 2), ("corr_gk_rv", "Gk", 2), ("ceil_oos_1h", "CeilOos", 2),
                        ("corr_lambda", "Lam", 2), ("slope_imb_on_bvc_1h", "Slope", 2)]:
        macro(f"panel{K}Med", fnum(t[col].median(), dgt))
        macro(f"panel{K}Min", fnum(t[col].min(), dgt))
        macro(f"panel{K}Max", fnum(t[col].max(), dgt))
    srt = t.sort_values("ceil_is_1h", ascending=False)
    macro("panelCeilMaxCoin", srt.symbol.iloc[0].replace("USDT", ""))
    macro("panelCeilSecond", fnum(srt.ceil_is_1h.iloc[1], 2))
    macro("panelCorrSecond", fnum(t.sort_values("corr_bvc_1h", ascending=False).corr_bvc_1h.iloc[1], 2))
    pos = t[t.fr_taker_t2 > 1.96].symbol.str.replace("USDT", "").tolist()
    macro("panelSigPosCoins", ", ".join(pos) if pos else "none")
    ratio = t.cs_median_bps / t.tick_bps
    macro("panelCsTickMin", f"{ratio.min():,.0f}"); macro("panelCsTickMax", f"{ratio.max():,.0f}")
    macro("panelCsTickMed", f"{ratio.median():,.0f}")
    macro("panelCeilAbove", str(int((t.ceil_is_1h > 0.5).sum())))
    for k in (1, 2, 4):
        w = HW[k]
        for tag, T in [("all_cfe", "All"), ("exbtc_cfe", "Ex"), ("all_tfe", "AllT"), ("exbtc_tfe", "ExT")]:
            macro(f"pool{T}B{w}", fnum(po[f"{tag}_k{k}_b"], 2))
            macro(f"pool{T}T{w}", fnum(po[f"{tag}_k{k}_t"], 1))
        macro(f"poolMkt{w}", fnum(po[f"decomp_b_market_k{k}"], 2)); macro(f"poolMktT{w}", fnum(po[f"decomp_t_market_k{k}"], 1))
        macro(f"poolCoin{w}", fnum(po[f"decomp_b_coin_k{k}"], 2)); macro(f"poolCoinT{w}", fnum(po[f"decomp_t_coin_k{k}"], 1))
        macro(f"panelNneg{w}", str(int(po[f"n_negative_{k}"])))
        macro(f"panelNsigNeg{w}", str(int(po[f"n_sig_negative_{k}"])))
        macro(f"panelNsigPos{w}", str(int(po[f"n_sig_positive_{k}"])))
        macro(f"panelSignP{w}", _p(po[f"sign_test_p_{k}"]))
    macro("panelSpSig", str(int(po["n_sp_taker_sig"])))
    macro("panelSpBvcSig", str(int(po["n_sp_bvc_sig"])))


def misc_macros():
    d = _res("BTC")
    if (d / "table_aggtrades_check.csv").exists():
        a = pd.read_csv(d / "table_aggtrades_check.csv").iloc[0]
        macro("aggHours", f"{int(a.n_hours):,}"); macro("aggCorr", f"{a.pearson:.4f}"); macro("aggMAD", f"{a.mean_abs_diff:.5f}")
    import json
    s = json.loads((d / "summary.json").read_text())
    macro("BTCnOOS", f"{s['n_oos']:,}"); macro("BTCoosStart", s["oos_start"][:10]); macro("BTCoosEnd", s["oos_end"][:10])
    macro("BTCnFoldsWF", str(s["n_folds"]))
    macro("nMSproxy", str(s["n_features"]["MS_proxy"])); macro("nMS", str(s["n_features"]["MS"]))
    macro("nTI", str(s["n_features"]["TI"])); macro("nALL", str(s["n_features"]["ALL"]))


def fig_cum_sse():
    from msr import plots
    oos = pd.read_csv(_res("BTC") / "oos_h4.csv.gz", index_col=0, parse_dates=True)
    ms = [m for m in ["enet_TI", "enet_MSproxy", "enet_MS", "enet_ALL", "enet_FLOWtrue", "gbm_ALL"] if f"f_{m}" in oos]
    (ROOT / "paper" / "figures").mkdir(exist_ok=True)
    plots.fig_cum_sse(oos, ms, ROOT / "paper" / "figures" / "fig_cum_sse.png")
    print("wrote fig_cum_sse.png")


def fig_fee_regime():
    """Weekly median hourly trade count, BTC/USDT vs ETH/USDT, zero-fee window shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from msr.data import load_ohlcv
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for sym, col in [("BTCUSDT", "#c4512b"), ("ETHUSDT", "#1f5fa8")]:
        p = ROOT / "data" / "raw" / f"{sym}_1h.csv"
        if p.exists():
            df = load_ohlcv(p, verbose=False)
            w = df["trades"].resample("W").median() / 1e3
            ax.plot(w.index, w.values, color=col, lw=1.1, label=sym.replace("USDT", "/USDT"))
    from msr.events import ZERO_FEE_OFF, ZERO_FEE_ON
    ax.axvspan(ZERO_FEE_ON, ZERO_FEE_OFF, color="0.85", lw=0, label="BTC zero-fee promotion")
    ax.set_yscale("log")
    ax.set_ylabel("trades per hour (000s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    (ROOT / "paper" / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "paper" / "figures" / "fig_fee_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_fee_regime.png")


if __name__ == "__main__":
    descriptive()
    zero_fee_activity()
    fig_fee_regime()
    fig_cum_sse()
    for a in ["BTC", "ETH"]:
        for fn in (validation_tables, forecast_tables):
            try:
                fn(a, main=(a == "BTC"))
            except FileNotFoundError as e:
                print("skip", fn.__name__, a, e)
        flow_response_tables(a)
    robustness_tables("BTC")
    response_split_table()
    single_predictor_table()
    single_trading_table()
    response_controls_table()
    ceiling_tables()
    fee_tables()
    fig_fee_copy()
    panel_tables()
    misc_macros()
    write_macros()
