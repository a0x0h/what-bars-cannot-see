"""Leakage-controlled walk-forward evaluation.

Design
------
* The out-of-sample (OOS) period is cut into consecutive blocks of `step_hours`.
* Before each block, every model is re-estimated on all data whose *label* is
  already observable at the block start: a training row t is kept only if
  t + h < block_start (purge). Nothing from the block is used.
* Hyper-parameters are selected inside each training window on its last
  `val_frac` share (itself purged from the inner training rows), then the model
  is refit on the whole training window.
* Winsorisation bounds and standardisation moments are estimated on the
  training window only (sklearn Pipeline).
* All models forecast the return relative to zero (no intercept / de-meaned
  target), the convention of Gu, Kelly & Xiu (2020). The historical-mean
  forecast is reported separately as a benchmark.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _prep():
    return [("winsor", Winsorizer()), ("scale", StandardScaler())]


# ----------------------------------------------------------------------------
# Model wrappers: fit(Xtr, ytr, Xval, yval) -> self, predict(X)
# ----------------------------------------------------------------------------
@dataclass
class ENetModel:
    """Elastic net with (alpha, l1_ratio) chosen on the inner validation block.

    The full alpha path is computed in one call per l1_ratio (warm starts), which
    is ~20x faster than refitting each grid point.
    """
    alphas: tuple = tuple(np.logspace(0, -4, 13))
    l1_ratios: tuple = (0.1, 0.5, 0.9)
    chosen_: dict = field(default_factory=dict)

    @staticmethod
    def _prep_fit(X):
        return Pipeline(_prep()).fit(X)

    def fit(self, Xtr, ytr, Xin, yin, Xval, yval):
        from sklearn.linear_model import enet_path
        sd = ytr.std()
        pin = self._prep_fit(Xin)
        Zin, Zval = pin.transform(Xin), pin.transform(Xval)
        best = (np.inf, None)
        for l1 in self.l1_ratios:
            alphas, coefs, _ = enet_path(Zin, yin / sd, l1_ratio=l1, alphas=np.array(self.alphas), max_iter=5000)
            pred = (Zval @ coefs) * sd  # (n_val, n_alphas)
            mse = ((yval[:, None] - pred) ** 2).mean(axis=0)
            j = int(np.argmin(mse))
            if mse[j] < best[0] - 1e-15:
                best = (mse[j], (float(alphas[j]), l1))
        a, l1 = best[1]
        self.chosen_ = {"alpha": a, "l1_ratio": l1}
        self.sd_ = sd
        self.pipe_ = Pipeline(_prep() + [("m", ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False,
                                                          max_iter=10000))]).fit(Xtr, ytr / sd)
        return self

    def predict(self, X):
        return self.pipe_.predict(X) * self.sd_

    def coefs(self, names):
        return pd.Series(self.pipe_.named_steps["m"].coef_, index=names)


@dataclass
class GBMModel:
    iters: tuple = (25, 50, 100, 200, 400)
    chosen_: dict = field(default_factory=dict)

    def _model(self, n):
        return HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=n, max_leaf_nodes=8, max_depth=3,
            min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=42)

    def fit(self, Xtr, ytr, Xin, yin, Xval, yval):
        mu_in, sd = yin.mean(), ytr.std()
        best = (np.inf, None)
        m = self._model(self.iters[0])
        m.set_params(warm_start=True)
        Zin = (yin - mu_in) / sd
        for n in self.iters:
            m.set_params(max_iter=n).fit(Xin, Zin)
            mse = np.mean((yval - m.predict(Xval) * sd) ** 2)
            if mse < best[0] - 1e-15:
                best = (mse, n)
        self.chosen_ = {"max_iter": best[1]}
        self.mu_, self.sd_ = ytr.mean(), sd
        self.m_ = self._model(best[1]).fit(Xtr, (ytr - self.mu_) / sd)
        return self

    def predict(self, X):
        return self.m_.predict(X) * self.sd_


MODEL_SPECS = {
    # name: (feature set, model class)
    "enet_TI": ("TI", ENetModel),
    "enet_MSproxy": ("MS_proxy", ENetModel),
    "enet_MS": ("MS", ENetModel),
    "enet_ALL": ("ALL", ENetModel),
    "enet_FLOWproxy": ("FLOW_proxy", ENetModel),
    "enet_FLOWtrue": ("FLOW_true", ENetModel),
    "gbm_MS": ("MS", GBMModel),
    "gbm_ALL": ("ALL", GBMModel),
}


def make_folds(index: pd.DatetimeIndex, h: int, min_train_hours: int, step_hours: int,
               window: str = "expanding", max_train_hours: int | None = None):
    """Yield (train_mask, test_mask, block_start) with purge t + h < block_start."""
    t0 = index[0] + pd.Timedelta(hours=min_train_hours)
    starts = pd.date_range(t0, index[-1], freq=f"{step_hours}h")
    folds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else index[-1] + pd.Timedelta(hours=1)
        test = (index >= s) & (index < e)
        cutoff = s - pd.Timedelta(hours=h)  # label of t uses close at t+h, need t+h < s
        train = index < cutoff
        if window == "rolling" and max_train_hours:
            train &= index >= cutoff - pd.Timedelta(hours=max_train_hours)
        if test.sum() and train.sum():
            folds.append((train, test, s))
    return folds


def inner_split(train_index: pd.DatetimeIndex, h: int, val_frac: float):
    n = len(train_index)
    vstart = train_index[int(n * (1 - val_frac))]
    inner = train_index < vstart - pd.Timedelta(hours=h)
    val = train_index >= vstart
    return inner, val


def walk_forward(X: pd.DataFrame, Y: pd.DataFrame, sets: dict, h: int, cfg: dict,
                 models: list[str] | None = None, verbose=True):
    """Run the walk-forward and return (oos_frame, coef_records, meta)."""
    ycol = f"y_{h}"
    ok = Y[ycol].notna()
    X, Y = X[ok], Y[ok]
    y = Y[ycol]
    models = models or list(MODEL_SPECS)
    folds = make_folds(X.index, h, cfg["min_train_hours"], cfg["step_hours"],
                       cfg.get("window", "expanding"), cfg.get("max_train_hours"))
    rows, coefs, chosen = [], [], []
    for k, (tr, te, s) in enumerate(folds):
        Xtr, ytr = X[tr], y[tr]
        inn, val = inner_split(Xtr.index, h, cfg["val_frac"])
        out = pd.DataFrame(index=X.index[te])
        out["y"] = y[te]
        out["r1_next"] = Y.loc[te, "r1_next"]
        out["fold"] = k
        out["f_zero"] = 0.0
        out["f_mean"] = ytr.mean()
        for name in models:
            fs, cls = MODEL_SPECS[name]
            cols = sets[fs]
            m = cls().fit(Xtr[cols].values, ytr.values, Xtr[cols].values[inn], ytr.values[inn],
                          Xtr[cols].values[val], ytr.values[val])
            out[f"f_{name}"] = m.predict(X.loc[te, cols].values)
            # threshold for the selective trading rule: 80th pct of |in-sample forecast|
            out[f"thr_{name}"] = np.quantile(np.abs(m.predict(Xtr[cols].values)), cfg.get("signal_quantile", 0.8))
            chosen.append({"fold": k, "model": name, **m.chosen_})
            if isinstance(m, ENetModel):
                c = m.coefs(cols)
                c.name = (k, name)
                coefs.append(c)
        rows.append(out)
        if verbose:
            print(f"  [wf h={h}] fold {k + 1}/{len(folds)} start={s.date()} train={tr.sum():,} test={te.sum():,}")
    oos = pd.concat(rows)
    coef_df = pd.DataFrame(coefs) if coefs else pd.DataFrame()
    meta = {"n_folds": len(folds), "oos_start": str(oos.index[0]), "oos_end": str(oos.index[-1]),
            "n_oos": int(len(oos)), "chosen": chosen}
    return oos, coef_df, meta
