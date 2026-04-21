# Crypto Alerter — Phase 2 (Rule Engine)

Implements the full gatekeeper logic from the Crypto Alerter v1 spec.
Builds on the phase-1 FVG detector by adding trend filters, momentum
triggers, volume confirmation, and SQLite alert logging.

## What it does

Every 5 minutes (Render cron), for BTC, ETH, and XRP:

1. Pulls OHLCV at 15m, 1h, 4h, daily, and 3-day from Coinbase public API.
2. Fetches the latest funding rate from Binance Futures (public, no key).
3. Detects FVGs with the 1.5x ATR-20 displacement filter.
4. Computes all indicators: 4h EMA(50), 15m RSI(14), 15m EMA(15)/EMA(50)
   histogram, 15m volume SMA(20), daily RSI(14), 3-day RSI(14), daily EMA(200).
5. Evaluates long and short setups against every active FVG.
6. Logs new alerts to SQLite with full context.
7. Prints everything to stdout (visible in Render logs).

## Long setup gates (all must pass)

| Gate | Condition |
|------|-----------|
| A | 4h close > 4h 50 EMA |
| B | Price tagging unfilled/tagged bullish FVG (4h or 1h), within 0.5% of far edge |
| C | 15m RSI was below 30 within last 5 candles, now above 30 |
| D | 15m EMA(15)-EMA(50) histogram flipped neg to pos within last 3 candles |
| E | Trigger candle volume >= 1.0x 20-period SMA |
| F | Stop ref logged: min(FVG low, recent 15m swing low) |
| G | Context logged: funding rate, session, distance from 4h EMA50, 4h RSI |

## Short setup gates (all must pass)

| Gate | Condition |
|------|-----------|
| A | Daily RSI > 70 OR 3-day RSI > 70 |
| B | 4h close < 4h 50 EMA AND EMA flat/down over 10 candles |
| C | Price tagging unfilled/tagged bearish FVG (4h or 1h), within 0.5% of near edge |
| D | 15m RSI was above 70 within last 5 candles, now below 70 |
| E | 15m EMA(15)-EMA(50) histogram flipped pos to neg within last 3 candles |
| F | Trigger candle volume >= 1.0x 20-period SMA |
| G | Funding rate <= +0.005% |
| H | Optional: price within 1% of daily 200 EMA boosts priority to high |
| I | Stop ref logged: max(FVG high, recent 15m swing high) |
| J | Context logged: daily RSI, 3-day RSI, funding, session, distance from 4h EMA50 |

## Layout

```
fvg-prototype/
├── src/
│   ├── coinbase_client.py     # Coinbase OHLCV pulls + 4h/3d resampling
│   ├── fvg_detector.py        # FVG detector + four-state model
│   ├── indicators.py          # RSI, EMA, histogram, volume, swing detection
│   ├── funding.py             # Binance funding rate (public API)
│   ├── alert_logger.py        # SQLite logging with full context
│   ├── rule_engine.py         # Gatekeeper logic for long + short setups
│   ├── run_alerter.py         # Phase 2 runner (what Render runs)
│   └── run_prototype.py       # Phase 1 runner (still works for FVG-only checks)
├── tests/
│   ├── test_detector.py       # 8 FVG state-machine tests
│   ├── test_indicators.py     # 14 indicator tests
│   └── test_rule_engine.py    # 6 rule engine tests
├── render.yaml
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt

# Run all tests (28 total)
PYTHONPATH=. python tests/test_detector.py
PYTHONPATH=. python tests/test_indicators.py
PYTHONPATH=. python tests/test_rule_engine.py

# Run the alerter (live data pull)
python -m src.run_alerter

# Run the phase 1 FVG-only prototype
python -m src.run_prototype
```

## Deploy to Render

1. Push to GitHub.
2. Render dashboard: New > Blueprint > point at the repo.
3. render.yaml creates a cron job running every 5 minutes.
4. Check Render logs for output. Alerts show as *** NEW ALERT lines.

## Reading the logs

Each run prints:
- Indicator snapshot per symbol: 4h close vs EMA50, 15m RSI, histogram
  value, volume ratio, daily/3d RSI, active FVG counts.
- *** NEW ALERT lines when all gates pass for a setup.
- Summary at the end: new alerts this run + total in DB.

When no gates pass (most runs), you see "no new alerts" per symbol.
That is expected — the spec is deliberately strict.

## SQLite notes

The alert DB lives at ./data/alerts.db (or ALERT_DB_PATH env var).
Each alert stores about 30 fields of context for post-hoc review.

On Render cron (no persistent disk), the DB resets on each deploy.
For the phase 2 paper run this is fine — alerts are in the logs.
For persistence, swap to Supabase in phase 5.

## What is next

- Phase 3: Backtest harness — run the rule engine against 6-12 months
  of historical data. Output MFE/MAE, hit rate, R:R distribution.
- Phase 4: Tune rules based on backtest results.
- Phase 5: Telegram bot + persistent storage.
