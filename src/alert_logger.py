"""
SQLite alert logger.

Every alert is stored with full context for post-hoc review.
Schema matches the spec's logged context requirements.

DB location controlled by ALERT_DB_PATH env var, defaults to ./data/alerts.db
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DB_PATH = Path(os.environ.get("ALERT_DB_PATH", "data/alerts.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,

    -- Core alert fields
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,        -- 'long' or 'short'
    setup_type TEXT NOT NULL,       -- 'day_long' or 'swing_short'
    priority TEXT DEFAULT 'normal', -- 'normal' or 'high' (daily 200 EMA confluence)

    -- FVG details
    fvg_tf TEXT NOT NULL,           -- timeframe of the FVG (1h or 4h)
    fvg_top REAL NOT NULL,
    fvg_bottom REAL NOT NULL,
    fvg_created_at TEXT NOT NULL,
    fvg_displacement_ratio REAL NOT NULL,

    -- Entry context
    entry_price REAL NOT NULL,      -- price at alert time (15m close)
    distance_into_fvg_pct REAL,     -- how far past the FVG edge

    -- Stop reference
    stop_ref REAL,                  -- logged, not a gate
    stop_source TEXT,               -- 'fvg_low', 'swing_low', 'fvg_high', 'swing_high'

    -- Gate conditions snapshot
    trend_4h_ema50 REAL,
    trend_4h_close REAL,
    trend_4h_ema50_slope TEXT,      -- 'up', 'flat', 'down' (short only)
    rsi_15m REAL,
    rsi_15m_bounced INTEGER,        -- 1/0
    ema_hist_15m REAL,
    ema_hist_flipped INTEGER,       -- 1/0
    volume_15m REAL,
    volume_sma_15m REAL,
    volume_ratio REAL,

    -- Logged context (not gates for longs)
    funding_rate_pct REAL,
    session TEXT,                   -- Asia/London/NY/Off
    distance_from_4h_ema50_pct REAL,
    rsi_4h REAL,

    -- Short-specific
    rsi_daily REAL,
    rsi_3d REAL,
    daily_200_ema REAL,
    distance_from_daily_200_pct REAL,

    -- Dedup
    alert_hash TEXT UNIQUE          -- prevent duplicate alerts for same setup
);

CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_direction ON alerts(direction);
"""


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def make_alert_hash(symbol: str, direction: str, fvg_created_at: str, fvg_tf: str) -> str:
    """
    Dedup hash: one alert per FVG per direction.
    If the same FVG triggers again on a later run, we skip it.
    """
    return f"{symbol}:{direction}:{fvg_tf}:{fvg_created_at}"


def alert_exists(alert_hash: str) -> bool:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM alerts WHERE alert_hash = ?", (alert_hash,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def log_alert(data: dict) -> bool:
    """
    Insert an alert record. Returns True if inserted, False if duplicate.
    `data` should contain all column names as keys.
    """
    conn = _get_conn()
    try:
        cols = [k for k in data.keys() if k != "id"]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [data[k] for k in cols]
        conn.execute(
            f"INSERT OR IGNORE INTO alerts ({col_names}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return conn.total_changes > 0
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_recent_alerts(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_alert_count() -> int:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
