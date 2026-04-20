"""
FVG (Fair Value Gap) detector.

Definition (three-candle model):
    BULLISH FVG: candle[i-1].high < candle[i+1].low
        Gap = [candle[i-1].high, candle[i+1].low]
        "Middle candle" = candle[i] (the displacement candle)

    BEARISH FVG: candle[i-1].low > candle[i+1].high
        Gap = [candle[i+1].high, candle[i-1].low]
        Middle candle = candle[i]

Displacement filter:
    Middle candle's range (high - low) must be >= displacement_mult * ATR(atr_period).
    Default spec: 1.5x ATR-20.

State model (per user spec):
    unfilled    — price has not entered the FVG since creation
    tagged      — price has entered the zone (wick or body) but has NOT closed
                  beyond the far side on the FVG's native timeframe.
                  This is the VALID ENTRY state for the long/short setups.
    filled      — a candle closed beyond the FVG's far side.
                    bullish: close < FVG top (the gap high)
                    bearish: close > FVG top (the gap low... wait, see below)
                  ^ careful: "far side" for a bullish FVG, from the perspective of
                  price coming down INTO it, is the top edge. A close below the top
                  means the gap is being filled / invalidated as support.
                  For bearish, far side from price coming UP is the bottom edge;
                  close above bottom means filled.

                  We use the user's rule: bullish filled = close below FVG TOP;
                  bearish filled = close above FVG BOTTOM.
                  (i.e. closed back into / through the gap from the reaction side)

    invalidated — price closed fully through the gap to the opposite side.
                    bullish: close < FVG bottom (swept through entirely)
                    bearish: close > FVG top
    stale       — `stale_after` candles elapsed on native TF with no tag.

All state transitions are determined on the FVG's NATIVE timeframe only.
A 4h FVG's state is only updated by 4h closes, never by 15m noise.

Outputs a list[FVG] sorted by creation time, newest last.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

import pandas as pd


class FVGDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGState(str, Enum):
    UNFILLED = "unfilled"
    TAGGED = "tagged"
    FILLED = "filled"
    INVALIDATED = "invalidated"
    STALE = "stale"


@dataclass
class FVG:
    symbol: str
    tf: str
    direction: FVGDirection
    created_at: pd.Timestamp       # timestamp of the middle (displacement) candle
    confirmed_at: pd.Timestamp     # timestamp of candle[i+1] — the one that confirmed the gap
    top: float                     # upper edge of the gap
    bottom: float                  # lower edge of the gap
    displacement_range: float      # high - low of the middle candle
    atr_at_creation: float
    displacement_ratio: float      # displacement_range / ATR
    state: FVGState = FVGState.UNFILLED
    first_tag_at: Optional[pd.Timestamp] = None
    filled_at: Optional[pd.Timestamp] = None
    invalidated_at: Optional[pd.Timestamp] = None
    candles_since_creation: int = 0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["state"] = self.state.value
        d["created_at"] = self.created_at.isoformat()
        d["confirmed_at"] = self.confirmed_at.isoformat()
        if self.first_tag_at is not None:
            d["first_tag_at"] = self.first_tag_at.isoformat()
        if self.filled_at is not None:
            d["filled_at"] = self.filled_at.isoformat()
        if self.invalidated_at is not None:
            d["invalidated_at"] = self.invalidated_at.isoformat()
        return d


def _atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Standard ATR using Wilder's smoothing via EMA."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing ~= EMA with alpha = 1/period
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def detect_fvgs(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    atr_period: int = 20,
    displacement_mult: float = 1.5,
    stale_after: int = 50,
) -> list[FVG]:
    """
    Detect FVGs in a DataFrame and evolve their state forward through the rest of the series.

    df must be indexed by timestamp and contain columns: open, high, low, close, volume.
    Returns a list of FVG objects with current state as of the last candle in df.
    """
    if len(df) < atr_period + 3:
        return []

    df = df.copy()
    df["atr"] = _atr(df, atr_period)

    fvgs: list[FVG] = []

    # Iterate candles i where we can look at [i-1, i, i+1]
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    opens = df["open"].values
    atrs = df["atr"].values
    index = df.index

    for i in range(1, len(df) - 1):
        atr_i = atrs[i]
        if pd.isna(atr_i) or atr_i <= 0:
            continue

        mid_range = highs[i] - lows[i]
        if mid_range < displacement_mult * atr_i:
            continue

        # Bullish FVG: gap between candle[i-1].high and candle[i+1].low
        if highs[i - 1] < lows[i + 1]:
            fvg = FVG(
                symbol=symbol,
                tf=tf,
                direction=FVGDirection.BULLISH,
                created_at=index[i],
                confirmed_at=index[i + 1],
                top=float(lows[i + 1]),
                bottom=float(highs[i - 1]),
                displacement_range=float(mid_range),
                atr_at_creation=float(atr_i),
                displacement_ratio=float(mid_range / atr_i),
            )
            fvgs.append(fvg)

        # Bearish FVG: gap between candle[i-1].low and candle[i+1].high
        elif lows[i - 1] > highs[i + 1]:
            fvg = FVG(
                symbol=symbol,
                tf=tf,
                direction=FVGDirection.BEARISH,
                created_at=index[i],
                confirmed_at=index[i + 1],
                top=float(lows[i - 1]),
                bottom=float(highs[i + 1]),
                displacement_range=float(mid_range),
                atr_at_creation=float(atr_i),
                displacement_ratio=float(mid_range / atr_i),
            )
            fvgs.append(fvg)

    # Evolve state forward for each FVG using candles after confirmed_at
    for fvg in fvgs:
        start_idx = df.index.get_loc(fvg.confirmed_at) + 1
        if start_idx >= len(df):
            continue

        for j in range(start_idx, len(df)):
            ts = index[j]
            h, l, c = highs[j], lows[j], closes[j]
            fvg.candles_since_creation = j - df.index.get_loc(fvg.created_at)

            if fvg.direction == FVGDirection.BULLISH:
                # Tag: any wick into [bottom, top]
                tagged_now = (l <= fvg.top) and (h >= fvg.bottom)

                # Invalidated: close fully below the gap bottom
                if c < fvg.bottom:
                    fvg.state = FVGState.INVALIDATED
                    fvg.invalidated_at = ts
                    if fvg.first_tag_at is None and tagged_now:
                        fvg.first_tag_at = ts
                    break

                # Filled: close below top (but not below bottom — that's invalidated above)
                if c < fvg.top:
                    if fvg.first_tag_at is None:
                        fvg.first_tag_at = ts
                    fvg.state = FVGState.FILLED
                    fvg.filled_at = ts
                    break

                if tagged_now and fvg.first_tag_at is None:
                    fvg.first_tag_at = ts
                    fvg.state = FVGState.TAGGED
                elif tagged_now and fvg.state == FVGState.UNFILLED:
                    fvg.state = FVGState.TAGGED

            else:  # BEARISH
                tagged_now = (h >= fvg.bottom) and (l <= fvg.top)

                # Invalidated: close fully above the gap top
                if c > fvg.top:
                    fvg.state = FVGState.INVALIDATED
                    fvg.invalidated_at = ts
                    if fvg.first_tag_at is None and tagged_now:
                        fvg.first_tag_at = ts
                    break

                # Filled: close above bottom (into the gap) but not above top
                if c > fvg.bottom:
                    if fvg.first_tag_at is None:
                        fvg.first_tag_at = ts
                    fvg.state = FVGState.FILLED
                    fvg.filled_at = ts
                    break

                if tagged_now and fvg.first_tag_at is None:
                    fvg.first_tag_at = ts
                    fvg.state = FVGState.TAGGED
                elif tagged_now and fvg.state == FVGState.UNFILLED:
                    fvg.state = FVGState.TAGGED

        # Stale check: unfilled / never-tagged AND older than stale_after
        if fvg.state == FVGState.UNFILLED and fvg.candles_since_creation >= stale_after:
            fvg.state = FVGState.STALE

    return fvgs


def fvgs_to_dataframe(fvgs: list[FVG]) -> pd.DataFrame:
    if not fvgs:
        return pd.DataFrame()
    return pd.DataFrame([f.to_dict() for f in fvgs])
