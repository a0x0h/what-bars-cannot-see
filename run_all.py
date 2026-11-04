"""Reproduce every table and figure of the paper.

    python run_all.py                       # uses the CSV(s) in data/raw
    python run_all.py --data path/to.csv    # explicit file
    python run_all.py --quick               # skip the slow robustness blocks

Outputs go to results/<ASSET>/ (CSV + LaTeX tables, PNG figures, summary.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from msr import analysis, backtest, inference, plots, validation  # noqa: E402
from msr.data import find_asset_files, find_companion, load_ohlcv  # noqa: E402
from msr.features import build_dataset  # noqa: E402
from msr.walkforward import MODEL_SPECS, walk_forward  # noqa: E402


def save(df: pd.DataFrame, out: Path, name: str, index=True, floatfmt="%.3f"):
    df.to_csv(out / f"{name}.csv", index=index)
    try:
        df.to_latex(out / f"{name}.tex", index=index, float_format=lambda x: floatfmt % x, na_rep="--")
    except Exception as e:  # jinja2 missing etc.
        print(f"  (latex export skipped for {name}: {e})")


def run_asset(tag: str, path: Path, cfg: dict, quick: bool, full: bool = True, validation_only: bool = False):
    t0 = time.time()
    out = ROOT / "results" / tag
    (out / "figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"])

    df = load_ohlcv(path)
    p1m = find_companion(path, "1m")
    m1 = load_ohlcv(p1m, freq="1min") if p1m else None
    H = cfg["main_horizon"]
    X, Y, sets, groups = build_dataset(df, cfg["horizons"], m1=m1)
    print(f"[{tag}] features: MS_proxy={len(sets['MS_proxy'])} MS={len(sets['MS'])} TI={len(sets['TI'])} "
          f"| rows after warm-up={len(X):,}")
    has_extra = len(sets["MS"]) > len(sets["MS_proxy"])
    core = ["enet_TI", "enet_MSproxy", "enet_MS", "enet_ALL"] if has_extra else ["enet_TI", "enet_MS", "enet_ALL"]
    flow = ["enet_FLOWproxy", "enet_FLOWtrue"] if "FLOW_true" in sets else []
    models = core + flow + (["gbm_MS", "gbm_ALL"] if full else [])

    # 0. Proxy validation against observed taker flow --------------------------
    val = validation.proxy_validation(df, X, m1)
    if not val.empty:
        save(val, out, "table_proxy_validation", index=False)
        save(validation.proxy_validation_by_year(df), out, "table_proxy_validation_by_year")
        save(validation.classifier_variants(df, m1), out, "table_classifier_variants", index=False)
        save(validation.information_ceiling(df, m1), out, "table_information_ceiling", index=False)
        save(validation.spread_proxy_diagnostics(df, X), out, "table_spread_diagnostics", index=False)
        vreg = validation.validation_by_regime(df, X)
        save(vreg, out, "table_validation_by_regime", index=False)
        plots.fig_bvc_calibration(df, m1, out / "figures" / "fig_bvc_calibration.png")
        plots.fig_validation_by_year(vreg, out / "figures" / "fig_validation_by_year.png")
        if "taker_imb_1" in X:
            fr = analysis.flow_response(df, X)
            save(fr, out, "table_flow_response", index=False)
            plots.fig_flow_response(fr, out / "figures" / "fig_flow_response.png")
            save(analysis.flow_response_splits(df, X), out, "table_flow_response_splits", index=False)
            from msr.single import single_predictor_oos
            sp = []
            for c in ["taker_imb_1", "ofi_1"]:
                t = single_predictor_oos(X[c], df, horizons=(1, 2, 4), min_train=cfg["walkforward"]["min_train_hours"],
                                         step=cfg["walkforward"]["step_hours"])
                t.insert(0, "predictor", c)
                sp.append(t)
            save(pd.concat(sp), out, "table_single_predictor", index=False)
            from msr import robust
            wf = cfg["walkforward"]
            save(robust.response_controls(df, X, m1), out, "table_response_controls", index=False)
            save(robust.single_predictor_extra(df, X, m1, min_train=wf["min_train_hours"], step=wf["step_hours"]),
                 out, "table_single_predictor_extra", index=False)
            save(robust.single_predictor_trading(df, X, min_train=wf["min_train_hours"], step=wf["step_hours"]),
                 out, "table_single_predictor_trading", index=False)
            save(robust.ceiling_sensitivity(df, m1), out, "table_ceiling_sensitivity", index=False)
            save(robust.ceiling_in_sample(df, m1), out, "table_ceiling_in_sample", index=False)
    pagg = find_companion(path, "agg")
    if pagg is not None and "taker_buy_volume" in df:
        save(validation.aggtrades_check(df, pd.read_csv(pagg, index_col=0)), out, "table_aggtrades_check", index=False)

    if validation_only:
        print(f"[{tag}] validation tables written to {out}")
        return {}

    # 1. Main walk-forward ----------------------------------------------------
    oos, coefs, meta = walk_forward(X, Y, sets, H, cfg["walkforward"], models=models)
    oos.to_csv(out / f"oos_h{H}.csv.gz")

    # 2. Forecast accuracy ----------------------------------------------------
    ftab = inference.forecast_table(oos, ["mean"] + models, H)
    save(ftab, out, "table_forecast_accuracy")
    lags = inference.nw_lags(H)
    incr = []
    for big, small in [("enet_ALL", "enet_TI"), ("enet_MS", "enet_TI"), ("enet_MS", "enet_MSproxy"),
                       ("enet_MSproxy", "enet_TI"), ("gbm_ALL", "enet_TI")]:
        if f"f_{big}" in oos and f"f_{small}" in oos:
            t, p = inference.clark_west(oos["y"], oos[f"f_{small}"], oos[f"f_{big}"], lags)
            dt, dp = inference.diebold_mariano(oos["y"], oos[f"f_{small}"], oos[f"f_{big}"], lags)
            incr.append({"model": big, "vs": small, "nested": True,
                         "R2oos_rel_%": 100 * inference.r2_oos(oos["y"], oos[f"f_{big}"], oos[f"f_{small}"]),
                         "CW_t": t, "CW_p": p, "DM_t": dt, "DM_p": dp})
    if "f_enet_FLOWtrue" in oos:
        # Not nested: observed vs proxy flow carry the same constructions on different inputs -> DM only
        dt, dp = inference.diebold_mariano(oos["y"], oos["f_enet_FLOWproxy"], oos["f_enet_FLOWtrue"], lags)
        incr.append({"model": "enet_FLOWtrue", "vs": "enet_FLOWproxy", "nested": False,
                     "R2oos_rel_%": 100 * inference.r2_oos(oos["y"], oos["f_enet_FLOWtrue"], oos["f_enet_FLOWproxy"]),
                     "CW_t": np.nan, "CW_p": np.nan, "DM_t": dt, "DM_p": dp})
    incr = pd.DataFrame(incr)
    save(incr, out, "table_incremental", index=False)

    # 3. Economic value ------------------------------------------------------
    bt, curves = backtest.backtest_table(oos, models, H, tuple(cfg["costs_bps"]))
    save(bt, out, "table_backtest", index=False, floatfmt="%.2f")
    save(backtest.strategy_diagnostics(oos, models, H, cost_bps=0), out, "table_strategy_diagnostics", index=False)

    # 4. Conditional predictability -----------------------------------------
    labels = analysis.regime_labels(X, oos.index)
    cond = analysis.conditional_table(oos, labels, core + flow, H)
    save(cond, out, "table_conditional", index=False)
    sub = analysis.subperiod_table(oos, core + flow, H)
    save(sub, out, "table_subperiods", index=False)

    # 5. Univariate evidence (in-sample, descriptive) ------------------------
    uni = analysis.univariate_table(X, Y, sets, groups, H)
    save(uni, out, "table_univariate", index=False)

    # 6. Figures ---------------------------------------------------------------
    fm = [m for m in ["enet_TI", "enet_MSproxy", "enet_MS", "enet_ALL", "gbm_ALL"] if m in models]
    plots.fig_cum_sse(oos, fm, out / "figures" / "fig_cum_sse.png")
    plots.fig_rolling_ic(oos, core, out / "figures" / "fig_rolling_ic.png")
    c2 = cfg["costs_bps"][1]
    plots.fig_equity(curves, [(m, "selective", c2) for m in fm] + [("buy_and_hold", "-", 0)],
                     out / "figures" / "fig_equity.png", f"Selective rule, net of {c2} bps per unit turnover")
    plots.fig_conditional(cond, "enet_ALL", out / "figures" / "fig_conditional.png")
    if not coefs.empty:
        plots.fig_coef_stability(coefs, "enet_ALL", out / "figures" / "fig_coef_stability.png")
        coefs.to_csv(out / "enet_coefficients_by_fold.csv")

    summary = {"asset": tag, "source": path.name, "n_bars": int(len(df)), "has_1m": m1 is not None,
               "feature_groups": groups,
               "missing_hours": int(df.attrs["n_missing_hours"]), "taker_flow": "taker_buy_volume" in df,
               "n_features": {k: len(v) for k, v in sets.items()}, "main_horizon": H, **meta,
               "forecast": ftab.round(5).reset_index().to_dict(orient="records"),
               "incremental": incr.round(5).to_dict(orient="records")}
    summary.pop("chosen", None)
    pd.DataFrame(meta["chosen"]).to_csv(out / "chosen_hyperparameters.csv", index=False)

    # 7. Robustness ------------------------------------------------------------
    if not quick:
        hz, _ = analysis.horizon_table(X, Y, sets, cfg["horizons"], cfg["walkforward"], models=core)
        save(hz, out, "table_horizons", index=False)
        abl = analysis.drop_group_ablation(X, Y, sets, groups, H, cfg["walkforward"])
        save(abl, out, "table_drop_group", index=False)
        wf_roll = dict(cfg["walkforward"], window="rolling", max_train_hours=cfg["rolling_train_hours"])
        oos_r, _, _ = walk_forward(X, Y, sets, H, wf_roll, models=core, verbose=False)
        rr = inference.forecast_table(oos_r, core, H)
        save(rr, out, "table_rolling_window")
        summary["robustness_done"] = True

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{tag}] done in {summary['runtime_sec']}s -> {out}")
    print(ftab.round(4).to_string())
    return summary


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="CSV file (default: every CSV in data/raw)")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--quick", action="store_true", help="skip horizon/ablation/rolling robustness")
    ap.add_argument("--validation-only", action="store_true", help="only the proxy-validation tables and figures")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    assets = {args.data.stem.upper(): args.data} if args.data else find_asset_files(ROOT / "data" / "raw")
    if not assets:
        sys.exit("No CSV found in data/raw. Put the hourly OHLCV file there or pass --data.")
    for i, (tag, p) in enumerate(assets.items()):
        # full model set and robustness for the primary asset; core models for the others
        run_asset(tag, p, cfg, quick=args.quick or i > 0, full=(i == 0), validation_only=args.validation_only)


if __name__ == "__main__":
    main()
