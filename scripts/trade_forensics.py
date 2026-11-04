"""Trade-level look at the volume that zero fees created (Binance aggregated trades).

    python scripts/trade_forensics.py

Downloads daily aggTrades files for BTC/USDT (treated) and ETH/USDT (control) on the two days
before and after each fee switch (about 0.5 GB, cached in data/raw/_cache/<SYMBOL>/aggTrades_daily)
and writes results/FEE/table_trade_forensics.csv with, per symbol and day:

    n_agg, n_fills        aggregated trades (one taker order at one price) and the fills they contain
    dvol_musd             dollar volume
    median_trade_usd      median dollar size of an aggregated trade
    pingpong_trades       share of aggregated trades that are followed (or preceded) within one second
                          by a trade of IDENTICAL quantity with the OPPOSITE aggressor side, i.e. an
                          immediate round trip of the same size
    pingpong_volume       the same share in dollar volume
    two_sided_1s          share of dollar volume matched by opposite aggressor flow within the same
                          second: 1 - sum_s |B_s - S_s| / sum_s (B_s + S_s)
    two_sided_1m          the same within the same minute
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download_binance import CACHE, fetch  # noqa: E402

BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
DAYS = {"fees (before removal)": ["2022-07-06", "2022-07-07"], "zero fees (after removal)": ["2022-07-09", "2022-07-10"],
        "zero fees (before restoration)": ["2023-03-20", "2023-03-21"], "fees (after restoration)": ["2023-03-23", "2023-03-24"]}
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "best_match"]


def load_day(symbol: str, day: str) -> pd.DataFrame | None:
    name = f"{symbol}-aggTrades-{day}.zip"
    z = CACHE / symbol / "aggTrades_daily" / name
    if not fetch(f"{BASE}/{symbol}/{name}", z):
        return None
    with zipfile.ZipFile(z) as zf:
        with zf.open(zf.namelist()[0]) as fi:
            has_header = not fi.readline().decode(errors="replace")[:1].isdigit()
        with zf.open(zf.namelist()[0]) as fi:
            d = pd.read_csv(fi, header=None, names=COLS, usecols=[1, 2, 3, 4, 5, 6], skiprows=int(has_header),
                            dtype={"price": float, "qty": float, "first_id": "int64", "last_id": "int64",
                                   "time": "int64", "is_buyer_maker": str})
    t = d["time"].to_numpy()
    d["time"] = np.where(t > 10 ** 14, t // 1000, t)  # microseconds from 2025 on
    d["side"] = np.where(d["is_buyer_maker"].str.lower().eq("true"), -1, 1)  # +1 buyer-initiated
    return d.drop(columns="is_buyer_maker")


def day_stats(d: pd.DataFrame) -> dict:
    qty, side, t, px = d["qty"].to_numpy(), d["side"].to_numpy(), d["time"].to_numpy(), d["price"].to_numpy()
    usd = qty * px
    # immediate round trips: next aggregated trade within 1 s, identical quantity, opposite side
    pair = (qty[1:] == qty[:-1]) & (side[1:] == -side[:-1]) & ((t[1:] - t[:-1]) <= 1000)
    in_pair = np.zeros(len(d), bool)
    in_pair[:-1] |= pair
    in_pair[1:] |= pair
    out = {"n_agg": len(d), "n_fills": int((d["last_id"] - d["first_id"] + 1).sum()), "dvol_musd": usd.sum() / 1e6,
           "median_trade_usd": float(np.median(usd)), "pingpong_trades": float(in_pair.mean()),
           "pingpong_volume": float(usd[in_pair].sum() / usd.sum())}
    for lab, ms in [("1s", 1000), ("1m", 60000)]:
        g = pd.DataFrame({"k": t // ms, "s": side * usd, "a": usd}).groupby("k").sum()
        out[f"two_sided_{lab}"] = float(1 - g["s"].abs().sum() / g["a"].sum())
    return out


def main():
    rows = []
    for regime, days in DAYS.items():
        for day in days:
            for sym in ["BTCUSDT", "ETHUSDT"]:
                d = load_day(sym, day)
                if d is None:
                    print(f"  missing {sym} {day}")
                    continue
                rows.append({"symbol": sym, "day": day, "regime": regime, **day_stats(d)})
                r = rows[-1]
                print(f"  {sym} {day} [{regime}]: {r['n_agg']:,} trades, ping-pong {r['pingpong_trades']:.1%} of trades, "
                      f"two-sided 1s {r['two_sided_1s']:.1%}", flush=True)
    out = ROOT / "results" / "FEE"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "table_trade_forensics.csv", index=False)
    print("wrote", out / "table_trade_forensics.csv")


if __name__ == "__main__":
    main()
