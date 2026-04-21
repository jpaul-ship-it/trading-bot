"""
Crypto Alerter — Main Runner (Phase 2)

Pulls all required timeframes for BTC/ETH/XRP, computes indicators,
runs the rule engine, and logs alerts to SQLite.

Every 5 minutes via Render cron:
  1. Fetch 15m, 1h, 4h, daily, 3d OHLCV from Coinbase
  2. Fetch funding rate from Binance
  3. Precompute all indicators
  4. Evaluate long + short setups
  5. Log new alerts to SQLite
  6. Print summary to stdout (visible in Render logs)

Usage:
    python -m src.run_alerter
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.coinbase_client import fetch_ohlcv
    from src.rule_engine import compute_market_data, evaluate_and_log
    from src.funding import funding_rate_pct
    from src.alert_logger import get_alert_count, get_recent_alerts
else:
    from .coinbase_client import fetch_ohlcv
    from .rule_engine import compute_market_data, evaluate_and_log
    from .funding import funding_rate_pct
    from .alert_logger import get_alert_count, get_recent_alerts


SYMBOLS = [
    ("BTC", "BTC-USD"),
    ("ETH", "ETH-USD"),
    ("XRP", "XRP-USD"),
]

# Candle counts per timeframe — enough history for all indicators
FETCH_CONFIG = {
    "15m": 500,    # ~5 days, enough for EMA(50) + RSI(14) + lookback
    "1h":  500,    # ~21 days, FVG detection on 1h
    "4h":  500,    # ~83 days, FVG + EMA(50) on 4h
    "1d":  300,    # ~300 days, RSI(14) + EMA(200) on daily
    "3d":  100,    # ~300 days, RSI(14) on 3-day
}


def fetch_all_timeframes(product_id: str, label: str) -> dict:
    """Fetch all required timeframes for a symbol. Returns dict of TF -> DataFrame."""
    frames = {}
    for tf, count in FETCH_CONFIG.items():
        try:
            df = fetch_ohlcv(product_id, tf, count)
            frames[tf] = df
            print(f"  [{label} {tf}] {len(df)} candles "
                  f"({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
        except Exception as e:
            print(f"  [{label} {tf}] FAILED: {e}")
            frames[tf] = pd.DataFrame()
    return frames


def run() -> None:
    run_start = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"CRYPTO ALERTER RUN  {run_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}")

    total_new_alerts = []

    for sym_label, product_id in SYMBOLS:
        print(f"\n--- {sym_label} ---")

        # 1. Fetch data
        frames = fetch_all_timeframes(product_id, sym_label)

        # Validate minimum data
        if frames.get("15m") is None or frames["15m"].empty:
            print(f"  ! skipping {sym_label}: no 15m data")
            continue
        if frames.get("4h") is None or frames["4h"].empty:
            print(f"  ! skipping {sym_label}: no 4h data")
            continue

        # 2. Fetch funding rate
        funding = None
        try:
            funding = funding_rate_pct(sym_label)
            print(f"  [funding] {funding:.4f}%" if funding is not None else "  [funding] unavailable")
        except Exception:
            print(f"  [funding] fetch failed")

        # 3. Compute market data bundle
        md = compute_market_data(
            symbol=sym_label,
            df_15m=frames["15m"],
            df_1h=frames.get("1h", pd.DataFrame()),
            df_4h=frames["4h"],
            df_daily=frames.get("1d"),
            df_3d=frames.get("3d"),
            funding_rate=funding,
        )

        # 4. Print current indicator snapshot
        if md.ema50_4h is not None and len(md.ema50_4h.dropna()) > 0:
            last_close = md.df_4h["close"].iloc[-1]
            last_ema = md.ema50_4h.iloc[-1]
            trend = "ABOVE" if last_close > last_ema else "BELOW"
            print(f"  [4h] close={last_close:.2f}  EMA50={last_ema:.2f}  {trend}")

        if md.rsi_15m is not None and len(md.rsi_15m.dropna()) > 0:
            print(f"  [15m] RSI={md.rsi_15m.iloc[-1]:.1f}  "
                  f"EMA_hist={md.ema_hist_15m.iloc[-1]:.4f}  "
                  f"vol_ratio={md.df_15m['volume'].iloc[-1] / md.vol_sma_15m.iloc[-1]:.2f}x"
                  if md.vol_sma_15m is not None and not pd.isna(md.vol_sma_15m.iloc[-1])
                  else f"  [15m] RSI={md.rsi_15m.iloc[-1]:.1f}")

        if md.rsi_daily is not None and len(md.rsi_daily.dropna()) > 0:
            print(f"  [daily] RSI={md.rsi_daily.iloc[-1]:.1f}")
        if md.rsi_3d is not None and len(md.rsi_3d.dropna()) > 0:
            print(f"  [3d] RSI={md.rsi_3d.iloc[-1]:.1f}")

        # Count active FVGs
        active_4h = len([f for f in (md.fvgs_4h or [])
                         if f.state.value in ("unfilled", "tagged")])
        active_1h = len([f for f in (md.fvgs_1h or [])
                         if f.state.value in ("unfilled", "tagged")])
        print(f"  [FVGs] 4h: {len(md.fvgs_4h or [])} total, {active_4h} active  "
              f"1h: {len(md.fvgs_1h or [])} total, {active_1h} active")

        # 5. Evaluate setups
        new_alerts = evaluate_and_log(md)
        total_new_alerts.extend(new_alerts)

        if not new_alerts:
            print(f"  no new alerts")

    # 6. Summary
    print(f"\n{'='*70}")
    print(f"RUN COMPLETE  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print(f"  new alerts this run: {len(total_new_alerts)}")
    print(f"  total alerts in DB:  {get_alert_count()}")

    if total_new_alerts:
        print(f"\n  NEW ALERTS:")
        for a in total_new_alerts:
            print(f"    {a['symbol']} {a['direction'].upper():5s} "
                  f"@ {a['entry_price']:<12.2f} "
                  f"FVG {a['fvg_tf']} [{a['fvg_bottom']:.2f}-{a['fvg_top']:.2f}]  "
                  f"priority={a['priority']}  "
                  f"session={a['session']}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
