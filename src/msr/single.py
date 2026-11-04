"""Out-of-sample test of a single pre-specified predictor (no tuning).

Expanding-window OLS without intercept on the standardised predictor (moments from the training
window), monthly refits, purged labels; evaluated with R2_OOS vs zero and Clark-West.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import clark_west, nw_lags, r2_oos
from .walkforward import make_folds


def single_predictor_oos(x: pd.Series, df: pd.DataFrame, horizons=(1, 2, 4), min_train=4380, step=720):
    lc = np.log(df["close"])
    rows = []
    for h in horizons:
        y = (lc.shift(-h) - lc).reindex(x.index)
        d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        preds = []
        for tr, te, _ in make_folds(d.index, h, min_train, step):
            a = d[tr]
            mu, sd = a["x"].mean(), a["x"].std()
            z = (a["x"] - mu) / sd
            b = float((z * a["y"]).sum() / (z * z).sum())
            preds.append(pd.Series(b * (d.loc[te, "x"] - mu) / sd, index=d.index[te]))
        f = pd.concat(preds)
        yy = d.loc[f.index, "y"]
        t, p = clark_west(yy, np.zeros(len(f)), f, nw_lags(h))
        rows.append({"horizon_h": h, "n_oos": len(f), "R2oos_%": 100 * r2_oos(yy, f, np.zeros(len(f))),
                     "CW_t": t, "CW_p": p, "IC": f.corr(yy, method="spearman")})
    return pd.DataFrame(rows)
