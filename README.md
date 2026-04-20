# FVG Prototype — Phase 1

Part of the Crypto Alerter v1 spec. This is the phase-1 deliverable: an
FVG detector, validated in isolation against real BTC/ETH/XRP data before
any rule engine is layered on top.

## What it does

1. Pulls OHLCV for BTC-USD, ETH-USD, XRP-USD at 15m, 1h, and 4h from
   Coinbase's public Exchange API (no key required).
2. Detects Fair Value Gaps using the three-candle model with a
   **1.5× ATR-20 displacement filter** on the middle candle.
3. Evolves each FVG's state forward through the data:
   - `unfilled` — price hasn't touched it
   - `tagged` — wick entered the zone, no close beyond far side *(this is the valid entry state for the setups in the spec)*
   - `filled` — closed beyond the far side (bullish: close below FVG top; bearish: close above FVG bottom)
   - `invalidated` — closed fully through the opposite side
   - `stale` — `stale_after` candles old without ever being tagged (default 50)
4. Dumps CSVs for eyeball cross-check against TradingView.

State transitions are always evaluated on the **FVG's native timeframe** —
a 4h FVG's state only changes on 4h closes, never on 15m noise.

## Layout

```
fvg-prototype/
├── src/
│   ├── coinbase_client.py   # Public OHLCV pulls + 4h/3d resampling
│   ├── fvg_detector.py      # Detector + state machine
│   └── run_prototype.py     # Phase-1 runner: pull, detect, dump CSVs
├── tests/
│   └── test_detector.py     # Synthetic state-machine tests (8 cases)
├── requirements.txt
├── render.yaml              # Render cron blueprint
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
PYTHONPATH=. python tests/test_detector.py     # sanity check (offline)
python -m src.run_prototype                    # live pull + detect
```

Outputs land in `./output/`:
- `candles_{SYMBOL}_{TF}.csv` — raw OHLCV
- `fvgs_{SYMBOL}_{TF}.csv` — detected FVGs with full state + timestamps
- `summary.csv` — counts per symbol / TF / state

## Deploy to Render

1. Push this directory to a GitHub repo.
2. In the Render dashboard: **New → Blueprint**, point at the repo.
   `render.yaml` defines a cron service running every 5 minutes.
3. Render will provision a 1 GB persistent disk at `/var/data` and the
   runner writes CSVs to `/var/data/fvg-output` (set via `FVG_OUTPUT_DIR`).
4. First run will cost ~15 seconds (9 TF fetches × ~1–2s each with polite
   throttling). Well under the 5-minute cron window.

Note: Render's free cron tier is fine for phase 1. You only need the
persistent disk once you want to inspect historical output across runs —
for pure eyeball-validation in phase 1 you could skip the disk entirely
and just read logs.

## Eyeball validation workflow

This is the part that matters for phase 1. Open TradingView alongside
the output and spot-check:

1. Pick 3–5 FVGs from `fvgs_BTC_4h.csv` — mix of `unfilled`, `tagged`,
   `filled`, `invalidated`.
2. For each one:
   - Jump to `created_at` on a BTC 4h TradingView chart.
   - Confirm the three-candle pattern exists (gap between candle N-1 and N+1, big middle candle).
   - Confirm the state matches: e.g. for a `filled` FVG, verify a later
     candle actually closed back through the top (bull) or bottom (bear).
3. Pay attention to **false negatives** — FVGs you can see on TV that the
   detector missed. Usually means the displacement filter is too strict,
   or the gap is very small. Log these; they inform tuning in phase 4.
4. Pay attention to **false positives** — FVGs flagged that aren't real
   structural levels. Often means the displacement candle was a wick, not
   a body. Could motivate a body-range filter in addition to H-L range.

Only when eyeball validation is clean do you move to phase 2 (rule engine).

## Parameters (all defaults match the v1 spec)

| Parameter           | Default | Location                     |
| ------------------- | ------- | ---------------------------- |
| `atr_period`        | 20      | `detect_fvgs()` arg          |
| `displacement_mult` | 1.5     | `detect_fvgs()` arg          |
| `stale_after`       | 50      | `detect_fvgs()` arg          |
| Candles per TF      | 500     | `run_prototype.CANDLES_PER_TF` |
| Fetch throttle      | 0.35s   | `coinbase_client.fetch_native` |

## Known limitations (phase 1 scope)

- No database yet. CSV is intentional — easy to diff, easy to eyeball. SQLite comes in phase 2 alongside the rule engine.
- No funding rate, no RSI, no EMAs. This module's only job is FVGs.
- 3-day and 4h are resampled from 1d and 1h respectively (Coinbase doesn't serve them natively). Resample boundaries use UTC, not exchange session time — TradingView's 4h candles on its default settings align with UTC too, so this matches.
- No alert pipeline. Alerts are phase 5.
