"""Data loading and validation for hourly OHLCV bars.

Accepts the common CSV layouts (Binance kline dumps, ccxt exports, TradingView,
generic Date/Open/High/Low/Close/Volume) and returns a clean, UTC-indexed,
strictly hourly DataFrame with columns: open, high, low, close, volume and,
when the source provides it, taker_buy_volume (aggressor-side buy volume).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_TIME_CANDIDATES = ["timestamp", "open_time", "opentime", "datetime", "date", "time", "unix", "open time"]
_TAKER_CANDIDATES = [
    "taker_buy_volume", "taker_buy_base_asset_volume", "taker buy base asset volume",
    "taker_buy_base_volume", "taker_base_vol", "takerbuybase",
]


def _find_col(cols, candidates):
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _parse_time(s: pd.Series) -> pd.DatetimeIndex:
    if np.issubdtype(s.dtype, np.number):
        v = s.astype("int64")
        # Detect unit from magnitude: s ~1e9, ms ~1e12, us ~1e15
        mag = np.nanmedian(np.abs(v.values))
        unit = "s" if mag < 1e11 else "ms" if mag < 1e14 else "us"
        return pd.to_datetime(v, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True)


def load_ohlcv(path: str | Path, verbose: bool = True, freq: str = "1h") -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path)
    # Binance raw dumps have no header: detect numeric first row
    if all(str(c).replace(".", "").isdigit() for c in raw.columns[:5]):
        raw = pd.read_csv(path, header=None)
        names = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
        raw.columns = names[: raw.shape[1]]

    tcol = _find_col(raw.columns, _TIME_CANDIDATES)
    if tcol is None:
        raise ValueError(f"No timestamp column found in {path.name}: {list(raw.columns)}")

    out = pd.DataFrame(index=_parse_time(raw[tcol]))
    for name in ["open", "high", "low", "close"]:
        c = _find_col(raw.columns, [name])
        if c is None:
            raise ValueError(f"Column '{name}' missing in {path.name}")
        out[name] = pd.to_numeric(raw[c], errors="coerce").values
    vcol = _find_col(raw.columns, ["volume", "volume btc", "volume_base", "vol", "base_volume"])
    if vcol is None:
        vcol = next((c for c in raw.columns if str(c).lower().startswith("volume")), None)
    if vcol is None:
        raise ValueError(f"No volume column in {path.name}")
    out["volume"] = pd.to_numeric(raw[vcol], errors="coerce").values
    tb = _find_col(raw.columns, _TAKER_CANDIDATES)
    if tb is not None:
        out["taker_buy_volume"] = pd.to_numeric(raw[tb], errors="coerce").values
    tc = _find_col(raw.columns, ["trades", "number_of_trades", "count", "trade_count", "tradecount"])
    if tc is not None:
        out["trades"] = pd.to_numeric(raw[tc], errors="coerce").values

    out.index.name = "time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])

    # Integrity checks
    bad = (out[["open", "high", "low", "close"]] <= 0).any(axis=1) | (out["low"] > out["high"]) \
        | (out["close"] > out["high"] * 1.0001) | (out["close"] < out["low"] * 0.9999)
    n_bad = int(bad.sum())
    out = out[~bad]

    full = pd.date_range(out.index[0], out.index[-1], freq=freq, tz="UTC")
    n_missing = len(full) - len(out)
    out = out.reindex(full)
    out.index.name = "time"
    if verbose:
        print(f"[data] {path.name}: {len(out):,} {freq} bars {out.index[0]} -> {out.index[-1]} | "
              f"invalid rows dropped={n_bad} | missing hours={n_missing} | "
              f"taker flow={'yes' if 'taker_buy_volume' in out else 'no'}")
    out.attrs["n_missing_hours"] = n_missing
    out.attrs["n_invalid_rows"] = n_bad
    out.attrs["source"] = path.name
    return out


def find_asset_files(raw_dir: str | Path) -> dict[str, Path]:
    """Map asset tag -> csv path for every CSV in data/raw (BTC first)."""
    files = [f for f in sorted(Path(raw_dir).glob("*.csv")) if "aggtrades" not in f.stem.lower()
             and not f.stem.lower().endswith("_1m")]
    res = {}
    for f in files:
        name = f.stem.upper()
        tag = "BTC" if "BTC" in name or "XBT" in name else "ETH" if "ETH" in name else f.stem
        res.setdefault(tag, f)
    return dict(sorted(res.items(), key=lambda kv: kv[0] != "BTC"))


def find_companion(path: Path, kind: str) -> Path | None:
    """For BTCUSDT_1h.csv find BTCUSDT_1m.csv(.gz) or BTCUSDT_aggtrades_hourly.csv."""
    stem = path.stem.replace("_1h", "")
    pats = {"1m": [f"{stem}_1m.csv.gz", f"{stem}_1m.csv"], "agg": [f"{stem}_aggtrades_hourly.csv"]}[kind]
    for p in pats:
        if (path.parent / p).exists():
            return path.parent / p
    return None
