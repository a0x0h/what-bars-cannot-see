"""Publication figures (matplotlib, grayscale-safe palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .inference import cumulative_sse_diff

PAL = {"enet_TI": "#9a9a9a", "enet_MSproxy": "#7fb3e0", "enet_MS": "#1f5fa8", "enet_ALL": "#c4512b",
       "gbm_MS": "#4c9a6a", "gbm_ALL": "#8a5fb0", "mean": "#555555", "buy_and_hold": "#222222"}
LABELS = {"enet_TI": "EN: TI", "enet_MSproxy": "EN: MS-proxy", "enet_MS": "EN: MS", "enet_ALL": "EN: ALL",
          "enet_FLOWproxy": "EN: Flow-BVC", "enet_FLOWtrue": "EN: Flow-taker", "gbm_MS": "GBM: MS",
          "gbm_ALL": "GBM: ALL", "buy_and_hold": "Buy and hold"}
PAL.update({"enet_FLOWproxy": "#e0a07f", "enet_FLOWtrue": "#2a9d8f"})
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _dates(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def fig_cum_sse(oos, models, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for m in models:
        ax.plot(cumulative_sse_diff(oos, m) * 1e4, label=LABELS.get(m, m), color=PAL.get(m), lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("cum. SSE(zero) - SSE(model)  (x 1e-4)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _dates(ax)
    fig.savefig(path)
    plt.close(fig)


def fig_rolling_ic(oos, models, path, window=720):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for m in models:
        ic = oos[f"f_{m}"].rolling(window).corr(oos["y"])
        ax.plot(ic, label=m, color=PAL.get(m), lw=1.1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Rolling 30-day IC (Pearson)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _dates(ax)
    fig.savefig(path)
    plt.close(fig)


def fig_equity(curves, keys, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for k in keys:
        if k in curves:
            ax.plot(curves[k], label=f"{k[0]} ({k[1]}, {k[2]} bps)" if k[1] != "-" else "buy & hold",
                    color=PAL.get(k[0]), lw=1.1)
    ax.set_ylabel("Equity (start = 1)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.5)
    _dates(ax)
    fig.savefig(path)
    plt.close(fig)


def fig_conditional(cond, model, path):
    sub = cond[(cond["model"] == model) & cond["dimension"].isin(["vol_regime", "illiq_regime", "session"])]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for ax, dim in zip(axes, ["vol_regime", "illiq_regime", "session"]):
        d = sub[sub["dimension"] == dim]
        order = {"low": 0, "mid": 1, "high": 2}
        d = d.sort_values("group", key=lambda s: s.map(order).fillna(s))
        ax.bar(d["group"], d["R2oos_%"], color=PAL.get(model, "#1f5fa8"))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(dim.replace("_", " "))
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    axes[0].set_ylabel("OOS R² vs zero (%)")
    fig.savefig(path)
    plt.close(fig)


def fig_coef_stability(coefs: pd.DataFrame, model, path, top=12):
    c = coefs[[ix[1] == model for ix in coefs.index]]
    if c.empty:
        return
    c.index = [ix[0] for ix in c.index]
    keep = c.abs().mean().sort_values(ascending=False).index[:top]
    c = c[keep]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * top + 1))
    v = np.nanmax(np.abs(c.values)) or 1
    im = ax.imshow(c.T.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(len(keep)), keep, fontsize=7)
    ax.set_xlabel("Re-estimation (monthly)")
    ax.set_title(f"{model}: standardised coefficients across refits")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(path)
    plt.close(fig)


def fig_bvc_calibration(df, m1, path, n_bins=40):
    """Observed taker imbalance vs standardised bar return, with the BVC mapping.

    Points: mean observed imbalance in quantile bins of z = r / sigma (with
    10-90% band of the within-bin distribution). Line: the BVC map 2*Phi(z) - 1.
    """
    from scipy.stats import norm
    panels = [("1-hour bars", df, 168)]
    if m1 is not None and "taker_buy_volume" in m1:
        panels.append(("1-minute bars", m1, 1440))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, d, w) in zip(axes, panels):
        r = np.log(d["close"]).diff()
        z = r / r.rolling(w, min_periods=w // 4).std()
        imb = (2 * d["taker_buy_volume"] - d["volume"]) / d["volume"]
        x = pd.DataFrame({"z": z, "imb": imb}).replace([np.inf, -np.inf], np.nan).dropna()
        x = x[x["z"].abs() < 6]
        b = pd.qcut(x["z"], n_bins, duplicates="drop")
        g = x.groupby(b, observed=True).agg(z=("z", "median"), m=("imb", "mean"),
                                           lo=("imb", lambda s: s.quantile(0.1)),
                                           hi=("imb", lambda s: s.quantile(0.9)))
        ax.fill_between(g["z"], g["lo"], g["hi"], color="#7fb3e0", alpha=0.35, lw=0, label="10-90% of observed")
        ax.plot(g["z"], g["m"], "o", ms=3, color="#1f5fa8", label="mean observed (binned)")
        zz = np.linspace(-4, 4, 200)
        ax.plot(zz, 2 * norm.cdf(zz) - 1, color="#c4512b", lw=1.3, label=r"BVC: $2\Phi(z)-1$")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_title(title)
        ax.set_xlabel(r"standardised bar return $z=r/\hat\sigma$")
    axes[0].set_ylabel("taker imbalance (2B - V)/V")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_validation_by_year(tab, path):
    t = tab[(tab["split"] == "year")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for agg, mk in [("1h", "o-"), ("24h", "s--")]:
        s = t[t["aggregation"] == agg]
        ax.plot(s["group"], s["pearson"], mk, label=f"corr, {agg}", color="#1f5fa8" if agg == "1h" else "#c4512b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("corr(BVC, observed)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


FEE_SHADE = "0.88"


def fig_switch_hours(win_on, win_off, path, n_controls=19):
    """Hourly BTC/USDT vs. control coins around the two fee switches."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    for j, (w, title) in enumerate([(win_on, "Fees removed: 8 Jul 2022 14:00 UTC"),
                                    (win_off, "Fees restored: 22 Mar 2023 00:00 UTC")]):
        ax = axes[0, j]
        ax.plot(w.index, w["trades_treated"], color="#c4512b", lw=1.1, label="BTC/USDT")
        ax.plot(w.index, w["trades_controls"], color="#1f5fa8", lw=1.1, label=f"{n_controls} control coins (mean)")
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(title, fontsize=8.5)
        ax = axes[1, j]
        ax.plot(w.index, w["abs_imb_treated_level"], color="#c4512b", lw=1.0)
        ax.plot(w.index, w["abs_imb_controls_level"], color="#1f5fa8", lw=1.0)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("hours from switch")
    axes[0, 0].set_ylabel("log trades\n(vs. 72h before)")
    axes[1, 0].set_ylabel("|taker imbalance|\n(6-hour average)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    for ax in axes.ravel():
        ax.set_xlim(-72, 72)
        ax.set_xticks([-72, -48, -24, 0, 24, 48, 72])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fee_event_study(es, keys_labels, on, off, path):
    """Weekly BTC-minus-controls differences relative to the pre-promotion mean (log points)."""
    n = len(keys_labels)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.75 * nrow + 0.4), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, lab) in zip(axes, keys_labels):
        ax.axvspan(on, off, color=FEE_SHADE, lw=0)
        ax.plot(es.index, es[k], color="#1f5fa8", lw=1.1, marker="o", ms=1.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(lab, fontsize=8.5)
        _dates(ax)
    for ax in axes[len(keys_labels):]:
        ax.set_visible(False)
    for i in range(0, len(keys_labels), ncol):
        axes[i].set_ylabel("BTC - controls\n(log, pre = 0)", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_btc_pairs(pdaily, events, path):
    """Daily dollar volume and volume-weighted |hourly imbalance| of Binance BTC/stablecoin pairs."""
    cols = {"BTC/USDT": "#c4512b", "BTC/BUSD": "#e0a07f", "BTC/TUSD": "#1f5fa8", "BTC/FDUSD": "#2a9d8f",
            "BTC/USDC": "#8a5fb0"}
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    for pair, g in pdaily.groupby("pair"):
        g = g.set_index("day").sort_index()
        g = g[(g.index >= pd.Timestamp("2022-01-01", tz="UTC")) & (g.index < pd.Timestamp("2024-07-01", tz="UTC"))]
        if g.empty:
            continue
        c = cols.get(pair, "0.4")
        axes[0].plot(g.index, (g["dvol"] / 1e6).rolling(7, min_periods=3).mean(), color=c, lw=1.1, label=pair)
        axes[1].plot(g.index, g["vw_abs_imb"].rolling(7, min_periods=3).mean(), color=c, lw=1.1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("dollar volume, $M/day\n(7-day mean, log)")
    axes[1].set_ylabel("|hourly taker imbalance|\n(volume-weighted, 7-day)")
    for ax in axes:
        for t, lab in events:
            ax.axvline(t, color="k", lw=0.6, ls="--")
        _dates(ax)
    ymax = axes[0].get_ylim()[1]
    for t, lab in events:
        axes[0].text(t, ymax, " " + lab, fontsize=6.5, va="top", ha="left", rotation=0)
    axes[0].legend(frameon=False, fontsize=7, ncol=5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_panel_ceiling(t, path):
    """Information ceiling and squared BVC correlation against trading volume, one point per coin."""
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = t["median_dvol_musd"]
    ax.scatter(x, t["ceil_is_1m"], s=16, color="#2a9d8f", label="ceiling, 1-minute bars", zorder=3)
    ax.scatter(x, t["ceil_is_1h"], s=16, color="#1f5fa8", label="ceiling, hourly bars", zorder=3)
    ax.scatter(x, t["corr2_bvc_1h"], s=16, marker="x", color="#c4512b", label="corr$^2$(BVC, observed), hourly",
               zorder=3)
    for _, r in t.iterrows():
        ax.annotate(r["symbol"].replace("USDT", ""), (r["median_dvol_musd"], r["ceil_is_1h"]), fontsize=6,
                    xytext=(2, -7), textcoords="offset points", color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("median hourly dollar volume ($M, log scale)")
    ax.set_ylabel("share of variance of observed\nhourly imbalance")
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_flow_response(tab, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for spec, reg, col, lab in [("taker", "taker", "#1f5fa8", "observed taker imbalance"),
                                ("bvc", "bvc", "#c4512b", "BVC imbalance (proxy)"),
                                ("joint", "taker", "#4c9a6a", "observed imbalance | bar return")]:
        s = tab[(tab.spec == spec) & (tab.regressor == reg)].sort_values("k")
        ax.plot(s["k"], s["beta_bps"], color=col, lw=1.4, label=lab)
        ax.fill_between(s["k"], s["beta_bps"] - 1.96 * s["se"], s["beta_bps"] + 1.96 * s["se"], color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("hours after bar $t$")
    ax.set_ylabel("cum. return per 1 SD of flow (bps)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)
