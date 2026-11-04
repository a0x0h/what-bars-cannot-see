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
        windows = {"Pre": ("2022-01-01", "2022-07-08"), "Zero": ("2022-07-08", "2023-03-22"),
                   "Post": ("2023-03-22", "2023-10-01")}
        for lab, (a, b) in windows.items():
            m = (t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))
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
        s = pd.read_csv(_res(asset) / "table_ceiling_sensitivity.csv")
        s1 = s[(s.bars == "1h") & (s.bins == 50)].set_index("split")
        macro(f"{asset}ceilRev", fnum(s1.loc["fit 2nd, eval 1st", "R2_ceiling"], 2))
    hdr = ("Period & Hours & Ceiling & corr$^2$(BVC) & Slope & SD imb. & Ceiling (1m)")
    write("tab_ceiling_periods", tab("lrrrrrr", hdr, L))


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
    ax.axvspan(pd.Timestamp("2022-07-08", tz="UTC"), pd.Timestamp("2023-03-22", tz="UTC"), color="0.85", lw=0,
               label="BTC zero-fee promotion")
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
    misc_macros()
    write_macros()
