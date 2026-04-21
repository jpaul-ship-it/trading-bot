"""
Rule Engine — Crypto Alerter v1

Implements all gatekeeper conditions from the frozen spec.

LONG SETUP (day-trading):
    A. 4h close > 4h 50 EMA
    B. Unfilled/tagged bullish FVG on 4h or 1h, price tagging, ≤0.5% past far edge
    C. 15m RSI(14) bounced above 30 within last 5 candles
    D. 15m EMA(15)-EMA(50) histogram flipped neg→pos within last 3 candles
    E. Trigger candle volume ≥ 1.0× 20-period SMA
    F. Stop ref = min(FVG low, recent 15m swing low)  [logged]
    G. Funding rate, session, distance from 4h 50 EMA, 4h RSI  [logged]

SHORT SETUP (swing, stricter):
    A. Daily RSI(14) > 70 OR 3-day RSI(14) > 70
    B. 4h close < 4h 50 EMA AND 4h 50 EMA flat/down over 10 candles
    C. Unfilled/tagged bearish FVG on 4h or 1h, price tagging from below, ≤0.5% past near edge
    D. 15m RSI(14) bounced below 70 within last 5 candles
    E. 15m EMA(15)-EMA(50) histogram flipped pos→neg within last 3 candles
    F. Trigger candle volume ≥ 1.0× 20-period SMA
    G. Funding rate ≤ +0.005%
    H. Daily 200 EMA proximity (optional confluence, boosts priority)
    I. Stop ref = max(FVG high, recent 15m swing high)  [logged]
    J. Daily RSI, 3-day RSI, funding, session, distance from 4h 50 EMA  [logged]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .fvg_detector import FVG, FVGDirection, FVGState, detect_fvgs
from .indicators import (
    rsi,
    ema,
    ema_histogram,
    volume_sma,
    ema_slope_flat_or_down,
    rsi_bounced_above,
    rsi_bounced_below,
    histogram_flipped_positive,
    histogram_flipped_negative,
    most_recent_swing_low,
    most_recent_swing_high,
    get_session,
)
from .funding import funding_rate_pct
from .alert_logger import log_alert, alert_exists, make_alert_hash


# ---------------------------------------------------------------------------
# Gate result (for debugging / logging which gates passed/failed)
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Precomputed data bundle — avoids recalculating indicators per FVG
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """All the indicator data needed by the rule engine, precomputed once per run."""
    symbol: str

    # DataFrames
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    df_4h: pd.DataFrame
    df_daily: Optional[pd.DataFrame] = None
    df_3d: Optional[pd.DataFrame] = None

    # 4h indicators
    ema50_4h: Optional[pd.Series] = None
    rsi_4h: Optional[pd.Series] = None

    # 15m indicators
    rsi_15m: Optional[pd.Series] = None
    ema_hist_15m: Optional[pd.Series] = None
    vol_sma_15m: Optional[pd.Series] = None

    # Daily / 3-day indicators (for short setup)
    rsi_daily: Optional[pd.Series] = None
    rsi_3d: Optional[pd.Series] = None
    ema200_daily: Optional[pd.Series] = None

    # FVGs
    fvgs_4h: Optional[list[FVG]] = None
    fvgs_1h: Optional[list[FVG]] = None

    # Funding rate
    funding_rate: Optional[float] = None


def compute_market_data(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    df_3d: Optional[pd.DataFrame] = None,
    funding_rate: Optional[float] = None,
) -> MarketData:
    """Precompute all indicators once per symbol per run."""
    md = MarketData(
        symbol=symbol,
        df_15m=df_15m,
        df_1h=df_1h,
        df_4h=df_4h,
        df_daily=df_daily,
        df_3d=df_3d,
    )

    # 4h
    if len(df_4h) >= 50:
        md.ema50_4h = ema(df_4h["close"], 50)
        md.rsi_4h = rsi(df_4h["close"], 14)

    # 15m
    if len(df_15m) >= 50:
        md.rsi_15m = rsi(df_15m["close"], 14)
        md.ema_hist_15m = ema_histogram(df_15m, fast=15, slow=50)
        md.vol_sma_15m = volume_sma(df_15m, 20)

    # Daily
    if df_daily is not None and len(df_daily) >= 200:
        md.rsi_daily = rsi(df_daily["close"], 14)
        md.ema200_daily = ema(df_daily["close"], 200)

    # 3-day
    if df_3d is not None and len(df_3d) >= 14:
        md.rsi_3d = rsi(df_3d["close"], 14)

    # FVGs
    md.fvgs_4h = detect_fvgs(df_4h, symbol, "4h")
    md.fvgs_1h = detect_fvgs(df_1h, symbol, "1h")

    md.funding_rate = funding_rate

    return md


# ---------------------------------------------------------------------------
# LONG SETUP evaluation
# ---------------------------------------------------------------------------

def _check_long_setup(md: MarketData) -> list[dict]:
    """
    Evaluate the long setup conditions against current market state.
    Returns a list of alert dicts (one per qualifying FVG).
    """
    alerts = []

    if md.ema50_4h is None or md.rsi_15m is None:
        return alerts

    # --- Gate A: 4h trend filter ---
    last_4h_close = md.df_4h["close"].iloc[-1]
    last_4h_ema50 = md.ema50_4h.iloc[-1]
    if pd.isna(last_4h_ema50) or last_4h_close <= last_4h_ema50:
        return alerts  # fail fast

    # --- Gate C: 15m RSI bounce above 30 ---
    if not rsi_bounced_above(md.rsi_15m, 30, lookback=5):
        return alerts

    # --- Gate D: 15m histogram flip neg→pos ---
    if not histogram_flipped_positive(md.ema_hist_15m, lookback=3):
        return alerts

    # --- Gate E: volume confirmation ---
    last_15m_vol = md.df_15m["volume"].iloc[-1]
    last_vol_sma = md.vol_sma_15m.iloc[-1]
    if pd.isna(last_vol_sma) or last_15m_vol < last_vol_sma:
        return alerts

    # --- Gate B: FVG structural level ---
    # Check both 4h and 1h FVGs
    current_price = md.df_15m["close"].iloc[-1]
    candidate_fvgs = []
    for fvg in (md.fvgs_4h or []) + (md.fvgs_1h or []):
        if fvg.direction != FVGDirection.BULLISH:
            continue
        if fvg.state not in (FVGState.UNFILLED, FVGState.TAGGED):
            continue

        # Price must be tagging the FVG: within or just past the zone
        # "No more than 0.5% past far edge" — for bullish FVG being tagged
        # from above, the far edge (bottom) is the lower bound.
        # Price tagging means price is inside or slightly below the gap.
        far_edge = fvg.bottom
        max_overshoot = far_edge * 0.005

        # Price is "tagging" if it's between fvg.bottom - overshoot and fvg.top
        if current_price <= fvg.top and current_price >= (far_edge - max_overshoot):
            candidate_fvgs.append(fvg)

    if not candidate_fvgs:
        return alerts

    # All gates passed — build alerts for each qualifying FVG
    for fvg in candidate_fvgs:
        alert_hash = make_alert_hash(md.symbol, "long", fvg.created_at.isoformat(), fvg.tf)
        if alert_exists(alert_hash):
            continue

        # Stop reference: lower of FVG low or recent 15m swing low
        swing_low = most_recent_swing_low(md.df_15m, len(md.df_15m) - 1)
        stop_ref = fvg.bottom
        stop_source = "fvg_low"
        if swing_low is not None and swing_low < fvg.bottom:
            stop_ref = swing_low
            stop_source = "swing_low"

        # Distance into FVG
        dist_into = 0.0
        if fvg.top != fvg.bottom:
            dist_into = (fvg.top - current_price) / (fvg.top - fvg.bottom) * 100

        # Distance from 4h EMA50
        dist_from_ema = ((current_price - last_4h_ema50) / last_4h_ema50) * 100

        alert_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": md.symbol,
            "direction": "long",
            "setup_type": "day_long",
            "priority": "normal",
            "fvg_tf": fvg.tf,
            "fvg_top": fvg.top,
            "fvg_bottom": fvg.bottom,
            "fvg_created_at": fvg.created_at.isoformat(),
            "fvg_displacement_ratio": round(fvg.displacement_ratio, 2),
            "entry_price": float(current_price),
            "distance_into_fvg_pct": round(dist_into, 2),
            "stop_ref": float(stop_ref),
            "stop_source": stop_source,
            "trend_4h_ema50": round(float(last_4h_ema50), 2),
            "trend_4h_close": round(float(last_4h_close), 2),
            "trend_4h_ema50_slope": None,
            "rsi_15m": round(float(md.rsi_15m.iloc[-1]), 2),
            "rsi_15m_bounced": 1,
            "ema_hist_15m": round(float(md.ema_hist_15m.iloc[-1]), 4),
            "ema_hist_flipped": 1,
            "volume_15m": float(last_15m_vol),
            "volume_sma_15m": round(float(last_vol_sma), 2),
            "volume_ratio": round(float(last_15m_vol / last_vol_sma), 2),
            "funding_rate_pct": md.funding_rate,
            "session": get_session(md.df_15m.index[-1]),
            "distance_from_4h_ema50_pct": round(dist_from_ema, 2),
            "rsi_4h": round(float(md.rsi_4h.iloc[-1]), 2) if md.rsi_4h is not None else None,
            "rsi_daily": None,
            "rsi_3d": None,
            "daily_200_ema": None,
            "distance_from_daily_200_pct": None,
            "alert_hash": alert_hash,
        }
        alerts.append(alert_data)

    return alerts


# ---------------------------------------------------------------------------
# SHORT SETUP evaluation
# ---------------------------------------------------------------------------

def _check_short_setup(md: MarketData) -> list[dict]:
    """
    Evaluate the short setup conditions against current market state.
    Returns a list of alert dicts (one per qualifying FVG).
    """
    alerts = []

    if md.ema50_4h is None or md.rsi_15m is None:
        return alerts

    # --- Gate A: Macro overbought ---
    daily_rsi_val = None
    three_day_rsi_val = None
    macro_overbought = False

    if md.rsi_daily is not None and len(md.rsi_daily) > 0:
        daily_rsi_val = float(md.rsi_daily.iloc[-1])
        if not pd.isna(daily_rsi_val) and daily_rsi_val > 70:
            macro_overbought = True

    if md.rsi_3d is not None and len(md.rsi_3d) > 0:
        three_day_rsi_val = float(md.rsi_3d.iloc[-1])
        if not pd.isna(three_day_rsi_val) and three_day_rsi_val > 70:
            macro_overbought = True

    if not macro_overbought:
        return alerts

    # --- Gate B: 4h trend filter (stricter) ---
    last_4h_close = md.df_4h["close"].iloc[-1]
    last_4h_ema50 = md.ema50_4h.iloc[-1]
    if pd.isna(last_4h_ema50) or last_4h_close >= last_4h_ema50:
        return alerts

    if not ema_slope_flat_or_down(md.ema50_4h, lookback=10):
        return alerts

    # --- Gate D: 15m RSI bounce below 70 ---
    if not rsi_bounced_below(md.rsi_15m, 70, lookback=5):
        return alerts

    # --- Gate E: 15m histogram flip pos→neg ---
    if not histogram_flipped_negative(md.ema_hist_15m, lookback=3):
        return alerts

    # --- Gate F: volume confirmation ---
    last_15m_vol = md.df_15m["volume"].iloc[-1]
    last_vol_sma = md.vol_sma_15m.iloc[-1]
    if pd.isna(last_vol_sma) or last_15m_vol < last_vol_sma:
        return alerts

    # --- Gate G: Funding rate ---
    if md.funding_rate is not None and md.funding_rate > 0.005:
        return alerts
    # If funding rate unavailable, we skip this gate but log it as None

    # --- Gate C: FVG structural level ---
    current_price = md.df_15m["close"].iloc[-1]
    candidate_fvgs = []
    for fvg in (md.fvgs_4h or []) + (md.fvgs_1h or []):
        if fvg.direction != FVGDirection.BEARISH:
            continue
        if fvg.state not in (FVGState.UNFILLED, FVGState.TAGGED):
            continue

        # For bearish FVG being tagged from below, "near edge" is the bottom.
        # "No more than 0.5% past near edge" means price can be up to 0.5% above the bottom.
        near_edge = fvg.bottom
        max_overshoot = near_edge * 0.005

        # Price is tagging if between fvg.bottom and fvg.top + overshoot
        if current_price >= fvg.bottom and current_price <= (fvg.top + max_overshoot):
            candidate_fvgs.append(fvg)

    if not candidate_fvgs:
        return alerts

    # --- Gate H: Daily 200 EMA proximity (optional, boosts priority) ---
    daily_200_val = None
    dist_from_200 = None
    near_daily_200 = False
    if md.ema200_daily is not None and len(md.ema200_daily) > 0:
        daily_200_val = float(md.ema200_daily.iloc[-1])
        if not pd.isna(daily_200_val) and daily_200_val > 0:
            dist_from_200 = abs(current_price - daily_200_val) / daily_200_val * 100
            near_daily_200 = dist_from_200 <= 1.0

    # Determine EMA slope label
    ema50_vals = md.ema50_4h.dropna()
    if len(ema50_vals) >= 10:
        start_v = ema50_vals.iloc[-10]
        end_v = ema50_vals.iloc[-1]
        if start_v > 0:
            chg = (end_v - start_v) / start_v
            if chg < -0.001:
                slope_label = "down"
            elif chg <= 0.001:
                slope_label = "flat"
            else:
                slope_label = "up"
        else:
            slope_label = "flat"
    else:
        slope_label = "unknown"

    # All gates passed — build alerts for each qualifying FVG
    for fvg in candidate_fvgs:
        alert_hash = make_alert_hash(md.symbol, "short", fvg.created_at.isoformat(), fvg.tf)
        if alert_exists(alert_hash):
            continue

        # Stop reference: higher of FVG high or recent 15m swing high
        swing_high = most_recent_swing_high(md.df_15m, len(md.df_15m) - 1)
        stop_ref = fvg.top
        stop_source = "fvg_high"
        if swing_high is not None and swing_high > fvg.top:
            stop_ref = swing_high
            stop_source = "swing_high"

        # Distance into FVG
        dist_into = 0.0
        if fvg.top != fvg.bottom:
            dist_into = (current_price - fvg.bottom) / (fvg.top - fvg.bottom) * 100

        # Distance from 4h EMA50
        dist_from_ema = ((current_price - float(last_4h_ema50)) / float(last_4h_ema50)) * 100

        priority = "high" if near_daily_200 else "normal"

        alert_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "symbol": md.symbol,
            "direction": "short",
            "setup_type": "swing_short",
            "priority": priority,
            "fvg_tf": fvg.tf,
            "fvg_top": fvg.top,
            "fvg_bottom": fvg.bottom,
            "fvg_created_at": fvg.created_at.isoformat(),
            "fvg_displacement_ratio": round(fvg.displacement_ratio, 2),
            "entry_price": float(current_price),
            "distance_into_fvg_pct": round(dist_into, 2),
            "stop_ref": float(stop_ref),
            "stop_source": stop_source,
            "trend_4h_ema50": round(float(last_4h_ema50), 2),
            "trend_4h_close": round(float(last_4h_close), 2),
            "trend_4h_ema50_slope": slope_label,
            "rsi_15m": round(float(md.rsi_15m.iloc[-1]), 2),
            "rsi_15m_bounced": 1,
            "ema_hist_15m": round(float(md.ema_hist_15m.iloc[-1]), 4),
            "ema_hist_flipped": 1,
            "volume_15m": float(last_15m_vol),
            "volume_sma_15m": round(float(last_vol_sma), 2),
            "volume_ratio": round(float(last_15m_vol / last_vol_sma), 2),
            "funding_rate_pct": md.funding_rate,
            "session": get_session(md.df_15m.index[-1]),
            "distance_from_4h_ema50_pct": round(dist_from_ema, 2),
            "rsi_4h": round(float(md.rsi_4h.iloc[-1]), 2) if md.rsi_4h is not None else None,
            "rsi_daily": round(daily_rsi_val, 2) if daily_rsi_val is not None else None,
            "rsi_3d": round(three_day_rsi_val, 2) if three_day_rsi_val is not None else None,
            "daily_200_ema": round(daily_200_val, 2) if daily_200_val is not None else None,
            "distance_from_daily_200_pct": round(dist_from_200, 2) if dist_from_200 is not None else None,
            "alert_hash": alert_hash,
        }
        alerts.append(alert_data)

    return alerts


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate_setups(md: MarketData) -> list[dict]:
    """
    Run both long and short setup checks.
    Returns list of alert dicts ready for logging and notification.
    """
    alerts = []
    alerts.extend(_check_long_setup(md))
    alerts.extend(_check_short_setup(md))
    return alerts


def evaluate_and_log(md: MarketData) -> list[dict]:
    """
    Evaluate setups and log any new alerts to SQLite.
    Returns list of newly logged alerts (excludes duplicates).
    """
    alerts = evaluate_setups(md)
    logged = []
    for alert in alerts:
        was_new = log_alert(alert)
        if was_new:
            logged.append(alert)
            print(f"  *** NEW ALERT: {alert['symbol']} {alert['direction'].upper()} "
                  f"@ {alert['entry_price']}  "
                  f"FVG [{alert['fvg_bottom']}-{alert['fvg_top']}] {alert['fvg_tf']}  "
                  f"session={alert['session']}  "
                  f"RSI={alert['rsi_15m']}  "
                  f"vol_ratio={alert['volume_ratio']}x")
    return logged
