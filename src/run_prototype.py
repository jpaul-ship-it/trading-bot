"""
Phase 1 prototype runner.

Pulls BTC/ETH/XRP at 15m and 4h from Coinbase, detects FVGs with the
1.5x ATR-20 displacement filter, and dumps:
  - output/candles_{SYMBOL}_{TF}.csv  — raw OHLCV for TradingView cross-check
  - output/fvgs_{SYMBOL}_{TF}.csv     — detected FVGs with full state
  - output/summary.csv                — counts per symbol/TF/state

Usage (local):
    python -m src.run_prototype
Usage (Render cron):
    Same command; schedule every 5 minutes or manual trigger.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python -m src.run_prototype` OR `python src/run_prototype.py`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.coinbase_client import fetch_ohlcv
    from src.fvg_detector import detect_fvgs, fvgs_to_dataframe
else:
    from .coinbase_client import fetch_ohlcv
    from .fvg_detector import detect_fvgs, fvgs_to_dataframe


SYMBOLS = [
    ("BTC", "BTC-USD"),
    ("ETH", "ETH-USD"),
    ("XRP", "XRP-USD"),
]

TIMEFRAMES = ["15m", "1h", "4h"]   # 1h added because spec allows FVGs on 1h too
CANDLES_PER_TF = {
    "15m": 500,   # ~5 days
    "1h":  500,   # ~21 days
    "4h":  500,   # ~83 days
}

OUTPUT_DIR = os.environ.get("FVG_OUTPUT_DIR", "")

# Active states worth seeing in the logs — skip stale/invalidated noise
ACTIVE_STATES = {"unfilled", "tagged"}


def run() -> None:
    # File output is optional — if FVG_OUTPUT_DIR is set, CSVs are written.
    # Either way, all active FVGs print to stdout (visible in Render logs).
    write_files = bool(OUTPUT_DIR)
    if write_files:
        out = Path(OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for sym_label, product_id in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"\n[{sym_label} {tf}] fetching {CANDLES_PER_TF[tf]} candles...")
            try:
                candles = fetch_ohlcv(product_id, tf, CANDLES_PER_TF[tf])
            except Exception as e:
                print(f"  ! fetch failed: {e}")
                continue

            if candles.empty:
                print(f"  ! no data returned")
                continue

            print(f"  got {len(candles)} candles  ({candles.index[0]} -> {candles.index[-1]})")

            if write_files:
                candles.to_csv(out / f"candles_{sym_label}_{tf}.csv")

            fvgs = detect_fvgs(
                candles,
                symbol=sym_label,
                tf=tf,
                atr_period=20,
                displacement_mult=1.5,
                stale_after=50,
            )
            fvg_df = fvgs_to_dataframe(fvgs)

            if write_files:
                fvg_df.to_csv(out / f"fvgs_{sym_label}_{tf}.csv", index=False)

            state_counts = (
                fvg_df["state"].value_counts().to_dict() if not fvg_df.empty else {}
            )
            print(f"  FVGs: {len(fvgs)} total  {state_counts}")

            # Print active FVGs (unfilled/tagged) for quick eyeball in logs
            if not fvg_df.empty:
                active = fvg_df[fvg_df["state"].isin(ACTIVE_STATES)]
                if not active.empty:
                    cols = ["direction", "state", "created_at", "top", "bottom",
                            "displacement_ratio", "first_tag_at"]
                    print(f"  ACTIVE ({len(active)}):")
                    for _, row in active.iterrows():
                        vals = "  ".join(f"{c}={row[c]}" for c in cols if c in row.index)
                        print(f"    {vals}")

            summary_rows.append({
                "symbol": sym_label,
                "tf": tf,
                "candles": len(candles),
                "fvgs_total": len(fvgs),
                "unfilled": state_counts.get("unfilled", 0),
                "tagged": state_counts.get("tagged", 0),
                "filled": state_counts.get("filled", 0),
                "invalidated": state_counts.get("invalidated", 0),
                "stale": state_counts.get("stale", 0),
                "last_candle": candles.index[-1].isoformat(),
            })

    summary_df = pd.DataFrame(summary_rows)
    if write_files:
        summary_df.to_csv(Path(OUTPUT_DIR) / "summary.csv", index=False)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
