"""Secondary analyses: conditional predictability, ablations, univariate evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import clark_west, ic_with_ci, nw_lags, predictive_regression, r2_oos
from .walkforward import walk_forward


def regime_labels(X: pd.DataFrame, index) -> pd.DataFrame:
    """Causal regime labels: volatility/liquidity tercile vs trailing 30 days, and session."""
    lab = pd.DataFrame(index=index)
    for col, name in [("gk_vol_24", "vol_regime"), ("amihud_24", "illiq_regime")]:
        x = X[col]
        pct = x.rolling(720, min_periods=240).rank(pct=True)
        lab[name] = pd.cut(pct.reindex(index), [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"])
    hr = index.hour
    lab["session"] = np.select([hr < 8, hr < 16], ["Asia 00-08", "Europe 08-16"], "US 16-24")
    lab["weekend"] = np.where(index.dayofweek >= 5, "weekend", "weekday")
    # Binance zero-fee promotion on BTC/USDT: 2022-07-08 to 2023-03-22 (calendar dates, known ex ante)
    zf = (index >= pd.Timestamp("2022-07-08", tz="UTC")) & (index < pd.Timestamp("2023-03-22", tz="UTC"))
    lab["fee_regime"] = np.where(zf, "zero-fee (BTC/USDT promo)", "standard fees")
    return lab


def conditional_table(oos: pd.DataFrame, labels: pd.DataFrame, models, h: int) -> pd.DataFrame:
    lags = nw_lags(h)
    rows = []
    for dim in labels.columns:
        for g, idx in oos.groupby(labels[dim].astype(str)).groups.items():
            if g == "nan":
                continue
            sub = oos.loc[idx]
            for m in models:
                f = sub[f"f_{m}"]
                t, p = clark_west(sub["y"], sub["f_zero"], f, lags)
                ic = f.corr(sub["y"], method="spearman")
                rows.append({"dimension": dim, "group": g, "model": m, "n": len(sub),
                             "R2oos_%": 100 * r2_oos(sub["y"], f, sub["f_zero"]), "IC": ic,
                             "CW_t": t, "CW_p": p})
    return pd.DataFrame(rows)


def drop_group_ablation(X, Y, sets, groups, h, cfg, base_model="enet_ALL"):
    """Re-run the full-feature ElasticNet without one microstructure group at a time."""
    from .walkforward import MODEL_SPECS
    fs_name = MODEL_SPECS[base_model][0]
    rows = []
    for gname, gcols in groups.items():
        sets2 = dict(sets)
        sets2[fs_name] = [c for c in sets[fs_name] if c not in gcols]
        oos, _, _ = walk_forward(X, Y, sets2, h, cfg, models=[base_model], verbose=False)
        rows.append({"dropped_group": gname, "n_features": len(sets2[fs_name]),
                     "R2oos_%": 100 * r2_oos(oos["y"], oos[f"f_{base_model}"], oos["f_zero"]),
                     "IC": oos[f"f_{base_model}"].corr(oos["y"], method="spearman")})
        print(f"  [ablation] without {gname}: R2oos={rows[-1]['R2oos_%']:.3f}%")
    return pd.DataFrame(rows)


def univariate_table(X: pd.DataFrame, Y: pd.DataFrame, sets, groups, h: int, n_sub: int = 4) -> pd.DataFrame:
    """Full-sample univariate predictive regressions (in-sample, descriptive) and
    sign stability across `n_sub` equal sub-periods."""
    y = Y[f"y_{h}"]
    lags = nw_lags(h)
    gmap = {c: g for g, cols in groups.items() for c in cols}
    gmap.update({c: "technical" for c in sets["TI"]})
    rows = []
    cuts = np.array_split(np.arange(len(X)), n_sub)
    for c in sets["ALL"]:
        b, t = predictive_regression(X[c], y, lags)
        signs = [np.sign(predictive_regression(X[c].iloc[ix], y.iloc[ix], lags)[0]) for ix in cuts]
        rows.append({"feature": c, "group": gmap[c], "beta_bps_per_sd": 1e4 * b, "t_NW": t,
                     "same_sign_subperiods": f"{int(sum(s == np.sign(b) for s in signs))}/{n_sub}"})
    return pd.DataFrame(rows).sort_values("t_NW", key=np.abs, ascending=False)


def horizon_table(X, Y, sets, horizons, cfg, models=("enet_TI", "enet_MS", "enet_ALL")):
    rows, store = [], {}
    for h in horizons:
        oos, _, _ = walk_forward(X, Y, sets, h, cfg, models=list(models), verbose=False)
        store[h] = oos
        for m in models:
            t, p = clark_west(oos["y"], oos["f_zero"], oos[f"f_{m}"], nw_lags(h))
            ic, (lo, hi) = ic_with_ci(oos["y"].values, oos[f"f_{m}"].values, n_boot=500)
            rows.append({"horizon_h": h, "model": m,
                         "R2oos_%": 100 * r2_oos(oos["y"], oos[f"f_{m}"], oos["f_zero"]),
                         "CW_t": t, "CW_p": p, "IC": ic, "IC_lo95": lo, "IC_hi95": hi})
        print(f"  [horizons] h={h} done")
    return pd.DataFrame(rows), store


def subperiod_table(oos: pd.DataFrame, models, h: int, n: int = 4):
    rows = []
    for k, ix in enumerate(np.array_split(np.arange(len(oos)), n)):
        sub = oos.iloc[ix]
        for m in models:
            t, p = clark_west(sub["y"], sub["f_zero"], sub[f"f_{m}"], nw_lags(h))
            rows.append({"subperiod": f"{sub.index[0].date()} to {sub.index[-1].date()}", "model": m,
                         "R2oos_%": 100 * r2_oos(sub["y"], sub[f"f_{m}"], sub["f_zero"]),
                         "IC": sub[f"f_{m}"].corr(sub["y"], method="spearman"), "CW_p": p,
                         "market_ret_%": 100 * (np.exp(np.log1p(sub["r1_next"]).sum()) - 1)})
    return pd.DataFrame(rows)


def flow_response(df: pd.DataFrame, X: pd.DataFrame, horizons=tuple(range(1, 25))) -> pd.DataFrame:
    """Cumulative return response to a one-SD order-flow shock (in-sample, HAC).

    For each k, regress log(C_{t+k}/C_t) (bps) on standardised regressors:
      taker   : observed taker imbalance of bar t
      bvc     : BVC imbalance of bar t
      joint   : taker imbalance and the bar's standardised return z_t jointly;
                the taker coefficient is then the effect of the part of flow
                that is not visible in the bar's own price change, i.e. the
                part no return-based classifier can recover.
    """
    import statsmodels.api as sm
    lc = np.log(df["close"])
    r = lc.diff()
    z = (r / r.rolling(168, min_periods=48).std()).rename("z")
    base = pd.concat([X["taker_imb_1"].rename("taker"), X["ofi_1"].rename("bvc"), z.reindex(X.index)], axis=1)
    rows = []
    for k in horizons:
        y = ((lc.shift(-k) - lc) * 1e4).reindex(X.index).rename("y")
        d = pd.concat([y, base], axis=1).dropna()
        sd = d[["taker", "bvc", "z"]].std()
        Z = (d[["taker", "bvc", "z"]] - d[["taker", "bvc", "z"]].mean()) / sd
        for spec, cols in [("taker", ["taker"]), ("bvc", ["bvc"]), ("joint", ["taker", "z"])]:
            res = sm.OLS(d["y"].values, sm.add_constant(Z[cols].values)).fit(
                cov_type="HAC", cov_kwds={"maxlags": max(24, 2 * k)})
            for j, c in enumerate(cols):
                rows.append({"k": k, "spec": spec, "regressor": c, "beta_bps": res.params[j + 1],
                             "se": res.bse[j + 1], "t": res.tvalues[j + 1], "r2_%": 100 * res.rsquared, "n": len(d)})
    return pd.DataFrame(rows)


def flow_response_splits(df: pd.DataFrame, X: pd.DataFrame, horizons=(1, 2, 4, 8)) -> pd.DataFrame:
    """Stability of the flow response across fee regimes and sample halves."""
    t = X.index
    zf = (t >= pd.Timestamp("2022-07-08", tz="UTC")) & (t < pd.Timestamp("2023-03-22", tz="UTC"))
    mid = pd.Timestamp("2024-07-01", tz="UTC")
    splits = {"Full sample": np.ones(len(t), bool), "Standard fees": ~zf, "Zero-fee window": zf,
              "Jan 2022 - Jun 2024": t < mid, "Jul 2024 - Oct 2026": t >= mid}
    out = []
    for name, m in splits.items():
        fr = flow_response(df, X[m], horizons=horizons)
        fr.insert(0, "sample", name)
        out.append(fr)
    return pd.concat(out, ignore_index=True)
