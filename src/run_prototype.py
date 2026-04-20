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

OUTPUT_DIR = Path(os.environ.get("FVG_OUTPUT_DIR", "output"))


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for sym_label, product_id in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"[{sym_label} {tf}] fetching {CANDLES_PER_TF[tf]} candles...")
            try:
                candles = fetch_ohlcv(product_id, tf, CANDLES_PER_TF[tf])
            except Exception as e:
                print(f"  ! fetch failed: {e}")
                continue

            if candles.empty:
                print(f"  ! no data returned")
                continue

            candles_out = OUTPUT_DIR / f"candles_{sym_label}_{tf}.csv"
            candles.to_csv(candles_out)
            print(f"  wrote {len(candles)} candles -> {candles_out}")

            fvgs = detect_fvgs(
                candles,
                symbol=sym_label,
                tf=tf,
                atr_period=20,
                displacement_mult=1.5,
                stale_after=50,
            )
            fvg_df = fvgs_to_dataframe(fvgs)
            fvg_out = OUTPUT_DIR / f"fvgs_{sym_label}_{tf}.csv"
            fvg_df.to_csv(fvg_out, index=False)

            state_counts = (
                fvg_df["state"].value_counts().to_dict() if not fvg_df.empty else {}
            )
            print(f"  detected {len(fvgs)} FVGs  states={state_counts}")

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
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nAll outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run()
