"""
Coinbase public API client for OHLCV data.

Uses the Exchange (formerly GDAX) public endpoint:
    https://api.exchange.coinbase.com/products/{product_id}/candles

No API key required. Returns up to 300 candles per request.

Granularities supported by Coinbase: 60, 300, 900, 3600, 21600, 86400 seconds.
    60     = 1m
    300    = 5m
    900    = 15m
    3600   = 1h
    21600  = 6h
    86400  = 1d

NOTE: Coinbase does NOT provide native 4h or 3d candles. Those must be
resampled from 1h and 1d respectively. See resample_ohlcv() below.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

COINBASE_BASE = "https://api.exchange.coinbase.com"

# Map our internal TF labels to Coinbase native granularities (seconds).
# None means "not native, must be resampled from a lower TF".
NATIVE_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
    "4h": None,   # resample from 1h
    "3d": None,   # resample from 1d
}

# Pandas resample rules for non-native TFs
RESAMPLE_RULE = {
    "4h": "4h",
    "3d": "3D",
}

# What we resample FROM when a TF is non-native
RESAMPLE_SOURCE = {
    "4h": "1h",
    "3d": "1d",
}


def _fetch_raw(product_id: str, granularity: int,
               start: datetime, end: datetime) -> list[list]:
    """Fetch raw candles from Coinbase. Returns list of [time, low, high, open, close, volume]."""
    url = f"{COINBASE_BASE}/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _to_dataframe(raw: list[list]) -> pd.DataFrame:
    """Coinbase returns [time, low, high, open, close, volume] in descending time order."""
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_native(product_id: str, tf: str, candles_wanted: int) -> pd.DataFrame:
    """
    Fetch `candles_wanted` candles at a native Coinbase granularity.
    Paginates backward from now in 300-candle chunks.
    """
    granularity = NATIVE_GRANULARITY[tf]
    if granularity is None:
        raise ValueError(f"{tf} is not a native Coinbase granularity — use fetch_ohlcv()")

    end = datetime.now(timezone.utc)
    chunk_size = 300  # Coinbase max per request
    chunk_seconds = granularity * chunk_size

    frames = []
    remaining = candles_wanted
    while remaining > 0:
        start = end - timedelta(seconds=chunk_seconds)
        raw = _fetch_raw(product_id, granularity, start, end)
        df = _to_dataframe(raw)
        if df.empty:
            break
        frames.append(df)
        remaining -= len(df)
        end = start
        time.sleep(0.35)  # stay polite to Coinbase public API

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.tail(candles_wanted)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a lower-TF OHLCV frame to a higher TF using standard OHLCV aggregation."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
    return out


def fetch_ohlcv(product_id: str, tf: str, candles_wanted: int = 500) -> pd.DataFrame:
    """
    Unified entry point. Handles native TFs directly and resamples 4h/3d.
    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume.
    """
    if tf not in NATIVE_GRANULARITY:
        raise ValueError(f"Unsupported timeframe: {tf}")

    if NATIVE_GRANULARITY[tf] is not None:
        return fetch_native(product_id, tf, candles_wanted)

    # Resample path — pull enough lower-TF candles to build `candles_wanted` upper-TF candles.
    source_tf = RESAMPLE_SOURCE[tf]
    # rough ratio to figure out how much source data we need
    ratio = {"4h": 4, "3d": 3}[tf]
    source_candles = candles_wanted * ratio + ratio  # buffer
    src_df = fetch_native(product_id, source_tf, source_candles)
    return resample_ohlcv(src_df, RESAMPLE_RULE[tf]).tail(candles_wanted)
