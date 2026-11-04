"""Download Binance public market data (data.binance.vision) into data/raw.

Run once on your own machine (standard library only, pandas optional):

    python scripts/download_binance.py                   # BTC + ETH, 1h + 1m, 2022-01 .. last full month
    python scripts/download_binance.py --start 2024-07   # shorter history
    python scripts/download_binance.py --aggtrades 2026-08 2026-09 2026-10   # + trade-level sample (needs pandas)

Outputs (data/raw):
    BTCUSDT_1h.csv        hourly klines incl. trade count and taker-buy volume
    BTCUSDT_1m.csv.gz     1-minute klines (same columns), gzip
    ETHUSDT_1h.csv / ETHUSDT_1m.csv.gz
    BTCUSDT_aggtrades_hourly.csv   (only with --aggtrades) hourly ground-truth flow stats

Notes
* Binance switched spot timestamps from milliseconds to microseconds on
  2025-01-01; everything here is normalised to milliseconds.
* Monthly files appear a few days after month end; missing months are skipped
  and reported.
* If the site does not open from your network, use a VPN and rerun: already
  downloaded zips are cached in data/raw/_cache and not fetched twice.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = RAW / "_cache"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "trades", "taker_buy_volume", "taker_buy_quote_volume"]


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def last_full_month() -> str:
    first = dt.date.today().replace(day=1)
    prev = first - dt.timedelta(days=1)
    return f"{prev.year:04d}-{prev.month:02d}"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "research-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(".part"), "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        dest.with_suffix(".part").rename(dest)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def to_ms(ts: str) -> int:
    v = int(float(ts))
    return v // 1000 if v > 10**14 else v


def download_klines(symbol: str, interval: str, start: str, end: str):
    out = RAW / (f"{symbol}_{interval}.csv" + (".gz" if interval == "1m" else ""))
    opener = gzip.open if interval == "1m" else open
    missing, n = [], 0
    with opener(out, "wt", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(KLINE_COLS)
        for mo in months(start, end):
            name = f"{symbol}-{interval}-{mo}.zip"
            z = CACHE / symbol / interval / name
            if not fetch(f"{BASE}/klines/{symbol}/{interval}/{name}", z):
                missing.append(mo)
                continue
            with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as fi:
                for row in csv.reader(io.TextIOWrapper(fi)):
                    if not row or not row[0][:1].isdigit():
                        continue  # header line in some files
                    row = row[:11]
                    row[0], row[6] = to_ms(row[0]), to_ms(row[6])
                    w.writerow(row)
                    n += 1
            print(f"  {symbol} {interval} {mo} ok", flush=True)
    print(f"[done] {out.name}: {n:,} rows" + (f" | missing months: {missing}" if missing else ""))


def download_aggtrades_hourly(symbol: str, mos: list[str]):
    """Aggregate trade-level data to hourly ground-truth order-flow statistics."""
    import numpy as np
    import pandas as pd

    cols = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker", "best_match"]
    parts = []
    for mo in mos:
        name = f"{symbol}-aggTrades-{mo}.zip"
        z = CACHE / symbol / "aggTrades" / name
        print(f"  {symbol} aggTrades {mo} downloading (large file)...", flush=True)
        if not fetch(f"{BASE}/aggTrades/{symbol}/{name}", z):
            print(f"  missing {mo}")
            continue
        with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as fi:
            for ch in pd.read_csv(fi, header=None, names=cols, usecols=[1, 2, 3, 4, 5, 6],
                                  chunksize=5_000_000, dtype={6: str}):
                ch = ch[pd.to_numeric(ch["time"], errors="coerce").notna()]
                t = ch["time"].astype("int64")
                t = np.where(t > 10**14, t // 1000, t)
                ch["hour"] = pd.to_datetime(t, unit="ms", utc=True).floor("h")
                ch["qty"] = ch["qty"].astype(float)
                ch["price"] = ch["price"].astype(float)
                sell = ch["is_buyer_maker"].astype(str).str.lower().eq("true")
                ch["buy_qty"] = np.where(sell, 0.0, ch["qty"])
                ch["sign_dollar"] = np.where(sell, -1.0, 1.0) * ch["qty"] * ch["price"]
                ch["dollar"] = ch["qty"] * ch["price"]
                ch["n_fills"] = ch["last_id"] - ch["first_id"] + 1
                g = ch.groupby("hour").agg(n_agg=("qty", "size"), n_fills=("n_fills", "sum"),
                                           volume=("qty", "sum"), buy_volume=("buy_qty", "sum"),
                                           dollar=("dollar", "sum"), signed_dollar=("sign_dollar", "sum"),
                                           max_trade_dollar=("dollar", "max"),
                                           p95_trade_dollar=("dollar", lambda s: s.quantile(0.95)))
                parts.append(g)
        print(f"  {symbol} aggTrades {mo} aggregated", flush=True)
    if not parts:
        return
    df = pd.concat(parts).groupby(level=0).agg({"n_agg": "sum", "n_fills": "sum", "volume": "sum",
                                                "buy_volume": "sum", "dollar": "sum", "signed_dollar": "sum",
                                                "max_trade_dollar": "max", "p95_trade_dollar": "max"})
    out = RAW / f"{symbol}_aggtrades_hourly.csv"
    df.to_csv(out, index_label="time")
    print(f"[done] {out.name}: {len(df):,} hours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", default=last_full_month())
    ap.add_argument("--no-1m", action="store_true")
    ap.add_argument("--aggtrades", nargs="*", default=[], help="months like 2026-09 (BTC only)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.symbols} {a.start}..{a.end} into {RAW}")
    try:
        for s in a.symbols:
            download_klines(s, "1h", a.start, a.end)
            if not a.no_1m:
                download_klines(s, "1m", a.start, a.end)
        if a.aggtrades:
            download_aggtrades_hourly(a.symbols[0], a.aggtrades)
    except urllib.error.URLError as e:
        sys.exit(f"\nNetwork error: {e}. data.binance.vision is not reachable from this network; "
                 f"turn on a VPN and rerun (finished files are cached).")


if __name__ == "__main__":
    main()
