"""
Technical indicators for the rule engine.

All functions take a pandas DataFrame with standard OHLCV columns
(open, high, low, close, volume) indexed by UTC timestamp.

Keeps pandas-ta out of the dependency tree — these are simple enough
to implement directly, and it means fewer failure modes on Render.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns Series aligned to input index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volume SMA
# ---------------------------------------------------------------------------

def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return df["volume"].rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# EMA histogram (fast - slow)
# ---------------------------------------------------------------------------

def ema_histogram(df: pd.DataFrame, fast: int = 15, slow: int = 50) -> pd.Series:
    """EMA(fast) - EMA(slow) on close prices."""
    return ema(df["close"], fast) - ema(df["close"], slow)


# ---------------------------------------------------------------------------
# Swing high / swing low detection
# ---------------------------------------------------------------------------

def swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Marks swing lows: candles where `low` is the minimum of the surrounding
    `lookback` candles on each side. Returns a boolean Series.
    """
    lows = df["low"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = lows.iloc[i - lookback: i + lookback + 1]
        if lows.iloc[i] == window.min():
            is_swing.iloc[i] = True
    return is_swing


def swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """
    Marks swing highs: candles where `high` is the maximum of the surrounding
    `lookback` candles on each side. Returns a boolean Series.
    """
    highs = df["high"]
    is_swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = highs.iloc[i - lookback: i + lookback + 1]
        if highs.iloc[i] == window.max():
            is_swing.iloc[i] = True
    return is_swing


def most_recent_swing_low(df: pd.DataFrame, as_of_idx: int, lookback: int = 5) -> Optional[float]:
    """Return the value of the most recent swing low at or before as_of_idx."""
    sls = swing_lows(df.iloc[:as_of_idx + 1], lookback)
    hits = sls[sls].index
    if len(hits) == 0:
        return None
    return float(df.loc[hits[-1], "low"])


def most_recent_swing_high(df: pd.DataFrame, as_of_idx: int, lookback: int = 5) -> Optional[float]:
    """Return the value of the most recent swing high at or before as_of_idx."""
    shs = swing_highs(df.iloc[:as_of_idx + 1], lookback)
    hits = shs[shs].index
    if len(hits) == 0:
        return None
    return float(df.loc[hits[-1], "high"])


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------

def get_session(ts: pd.Timestamp) -> str:
    """
    Classify a UTC timestamp into a trading session.
    Asia:   00:00 - 08:00 UTC
    London: 08:00 - 13:00 UTC
    NY:     13:00 - 21:00 UTC
    Off:    21:00 - 00:00 UTC
    """
    hour = ts.hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "NY"
    else:
        return "Off"


# ---------------------------------------------------------------------------
# EMA slope (for short setup condition B)
# ---------------------------------------------------------------------------

def ema_slope_flat_or_down(ema_series: pd.Series, lookback: int = 10) -> bool:
    """
    Check if the EMA is flat or down-sloping over the last `lookback` candles.
    "Flat" = change over the period is within +0.1% of the starting value.
    "Down" = ending value < starting value.
    """
    if len(ema_series) < lookback:
        return False
    recent = ema_series.iloc[-lookback:]
    start_val = recent.iloc[0]
    end_val = recent.iloc[-1]
    if start_val == 0:
        return False
    pct_change = (end_val - start_val) / start_val
    return pct_change <= 0.001  # flat threshold: +0.1%


# ---------------------------------------------------------------------------
# RSI bounce detection (for momentum triggers)
# ---------------------------------------------------------------------------

def rsi_bounced_above(rsi_series: pd.Series, level: float, lookback: int = 5) -> bool:
    """
    Check if RSI was below `level` within the last `lookback` candles
    AND the current candle's RSI is above `level`.

    Used for long setup: RSI < 30 then back above 30.
    """
    if len(rsi_series) < lookback + 1:
        return False
    current = rsi_series.iloc[-1]
    recent = rsi_series.iloc[-(lookback + 1):-1]
    was_below = (recent < level).any()
    now_above = current > level
    return bool(was_below and now_above)


def rsi_bounced_below(rsi_series: pd.Series, level: float, lookback: int = 5) -> bool:
    """
    Check if RSI was above `level` within the last `lookback` candles
    AND the current candle's RSI is below `level`.

    Used for short setup: RSI > 70 then back below 70.
    """
    if len(rsi_series) < lookback + 1:
        return False
    current = rsi_series.iloc[-1]
    recent = rsi_series.iloc[-(lookback + 1):-1]
    was_above = (recent > level).any()
    now_below = current < level
    return bool(was_above and now_below)


# ---------------------------------------------------------------------------
# EMA histogram flip detection
# ---------------------------------------------------------------------------

def histogram_flipped_positive(hist: pd.Series, lookback: int = 3) -> bool:
    """
    Check if EMA histogram flipped from negative to positive within
    the last `lookback` candles. Current value must be positive.
    """
    if len(hist) < lookback + 1:
        return False
    current = hist.iloc[-1]
    recent = hist.iloc[-(lookback + 1):-1]
    was_negative = (recent < 0).any()
    now_positive = current > 0
    return bool(was_negative and now_positive)


def histogram_flipped_negative(hist: pd.Series, lookback: int = 3) -> bool:
    """
    Check if EMA histogram flipped from positive to negative within
    the last `lookback` candles. Current value must be negative.
    """
    if len(hist) < lookback + 1:
        return False
    current = hist.iloc[-1]
    recent = hist.iloc[-(lookback + 1):-1]
    was_positive = (recent > 0).any()
    now_negative = current < 0
    return bool(was_positive and now_negative)
