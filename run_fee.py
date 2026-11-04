"""Zero-fee natural experiment: tables and figures for Section 5 of the paper.

    python scripts/download_binance.py --outdir data/raw/panel --symbols BTCUSDT ETHUSDT ...   (see README)
    python scripts/download_binance.py --outdir data/raw/pairs --no-1m --symbols BTCBUSD BTCTUSD BTCFDUSD BTCUSDC
    python run_fee.py

Treated unit: BTC/USDT. Controls: every other USDT pair in data/raw/panel (standard fees throughout).
Outputs: results/FEE/ (CSV tables, figures, daily measures per coin in results/FEE/daily).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from msr import fee, plots  # noqa: E402
from msr.events import TUSD_OFF, ZERO_FEE_OFF, ZERO_FEE_ON  # noqa: E402

PANEL = ROOT / "data" / "raw" / "panel"
PAIRS = ROOT / "data" / "raw" / "pairs"
OUT = ROOT / "results" / "FEE"
TREATED = "BTCUSDT"


def daily_for(sym: str) -> pd.DataFrame:
    src = PANEL / f"{sym}_1m.csv.gz"
    cache = OUT / "daily" / f"{sym}.csv.gz"
    if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
        d = pd.read_csv(cache, index_col=0)
        d.index = pd.to_datetime(d.index, utc=True)
        return d
    t = time.time()
    d = fee.daily_measures(fee.load_klines(src, "1min"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(cache)
    print(f"  {sym}: {len(d)} days ({time.time() - t:.0f}s)", flush=True)
    return d


def decomposition(panel, controls):
    """Excess BTC volume during the promotion relative to the control-implied counterfactual, split
    into directional and two-sided volume (ratio difference-in-differences in levels)."""
    rows = {}
    for col in ["dvol", "net1m", "two_sided", "trades"]:
        Y = pd.DataFrame({s: d[col] for s, d in panel.items()})
        mean = lambda win: Y[(Y.index >= win[0]) & (Y.index < win[1])].mean()  # noqa: E731
        pre, zero = mean(fee.PRE), mean(fee.ZERO)
        growth_c = np.exp(np.log(zero[controls] / pre[controls]).mean())  # geometric mean control growth
        cf = pre[TREATED] * growth_c
        rows[col] = {"btc_pre": pre[TREATED], "btc_zero": zero[TREATED], "control_growth": growth_c,
                     "counterfactual": cf, "excess": zero[TREATED] - cf}
    t = pd.DataFrame(rows).T
    t["share_of_excess_dollar_volume"] = t["excess"] / t.loc["dvol", "excess"]
    return t


def main():
    t0 = time.time()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    syms = sorted(p.name.replace("_1m.csv.gz", "") for p in PANEL.glob("*_1m.csv.gz"))
    if len(sys.argv) > 1:  # optional explicit list, e.g. while downloads are still running
        syms = [s for s in syms if s in sys.argv[1:]]
    if TREATED not in syms or len(syms) < 4:
        sys.exit("run the panel download first (see the docstring)")
    controls = [s for s in syms if s != TREATED]
    print(f"[fee] treated {TREATED}, {len(controls)} controls: {', '.join(c.replace('USDT', '') for c in controls)}")
    panel = {s: daily_for(s) for s in syms}

    # 1. Difference-in-differences ---------------------------------------------------------
    did = fee.did_table(panel, TREATED, controls)
    did.to_csv(OUT / "table_did.csv", index=False)
    fee.did_table(panel, TREATED, ["ETHUSDT"]).to_csv(OUT / "table_did_eth.csv", index=False)
    fee.did_table(panel, TREATED, controls, days=60).to_csv(OUT / "table_did_60d.csv", index=False)
    fee.did_table(panel, TREATED, controls, days=180).to_csv(OUT / "table_did_180d.csv", index=False)
    show = did[["outcome", "on", "on_t", "p_space_on", "p_time_on", "off", "off_t", "p_space_off", "p_time_off",
                "on_full", "off_full", "on_trend", "off_trend"]]
    print(show.round(3).to_string())

    dec = decomposition(panel, controls)
    dec.to_csv(OUT / "table_decomposition.csv")
    print(dec.round(3).to_string())

    # 2. Event study (weekly) ---------------------------------------------------------------
    keys = ["trades", "dvol", "net1m", "two_sided", "amihud", "lam_bvc", "lam_true", "abs_imb_h"]
    es = fee.event_study(panel, TREATED, controls, keys)
    es.to_csv(OUT / "event_study_weekly.csv")
    plots.fig_fee_event_study(es, [(k, fee.OUTCOMES[k][2]) for k in keys], ZERO_FEE_ON, ZERO_FEE_OFF,
                              OUT / "figures" / "fig_fee_event_study.png")

    # 3. Hour-level switches ------------------------------------------------------------------
    hourly = {s: fee.hourly_measures(fee.load_klines(PANEL / f"{s}_1h.csv", "h")) for s in syms}
    w_on = fee.switch_window(hourly, TREATED, controls, ZERO_FEE_ON)
    w_off = fee.switch_window(hourly, TREATED, controls, ZERO_FEE_OFF)
    w_on.to_csv(OUT / "switch_window_on.csv")
    w_off.to_csv(OUT / "switch_window_off.csv")
    plots.fig_switch_hours(w_on, w_off, OUT / "figures" / "fig_switch_hours.png", n_controls=len(controls))
    jumps = {"on": fee.switch_jumps(hourly, TREATED, controls, ZERO_FEE_ON),
             "off": fee.switch_jumps(hourly, TREATED, controls, ZERO_FEE_OFF)}
    # Same 24h jump at hours spread over 2024-2026 as a placebo distribution for the hourly break
    for k in ["trades", "abs_imb"]:
        D = fee.hourly_diff(hourly, TREATED, controls, k)
        pl = np.array([fee.hour_jump(D, t) for t in pd.date_range("2024-01-02", "2026-10-30", freq="25h", tz="UTC")])
        pl = np.abs(pl[np.isfinite(pl)])
        jumps[f"placebo_{k}_abs_p99"] = float(np.quantile(pl, 0.99))
        jumps[f"placebo_{k}_abs_max"] = float(pl.max())
        jumps[f"n_placebo_hours_{k}"] = int(len(pl))
    (OUT / "switch_jumps.json").write_text(json.dumps(jumps, indent=2))
    print("[fee] 24h switch jumps:", json.dumps(jumps))

    # 4. Where the fee-free volume went: other BTC pairs -------------------------------------
    pairs = {"BTC/USDT": PANEL / "BTCUSDT_1h.csv"}
    for p in ["BUSD", "TUSD", "FDUSD", "USDC"]:
        f = PAIRS / f"BTC{p}_1h.csv"
        if f.exists():
            pairs[f"BTC/{p}"] = f
    pdaily = fee.pair_daily({k: fee.load_klines(v, "h") for k, v in pairs.items()})
    pdaily.to_csv(OUT / "btc_pairs_daily.csv", index=False)
    preg = fee.pair_regime_table(pdaily)
    preg.to_csv(OUT / "table_btc_pairs.csv", index=False)
    print(preg.round(3).to_string())
    events = [(ZERO_FEE_ON, "13 BTC pairs fee-free"), (ZERO_FEE_OFF, "only TUSD fee-free"),
              (TUSD_OFF, "TUSD fees")]
    plots.fig_btc_pairs(pdaily, events, OUT / "figures" / "fig_btc_pairs.png")
    print(f"[fee] done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
