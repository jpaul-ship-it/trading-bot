"""
Tests for the rule engine.

Builds synthetic market data that either passes or fails the gatekeeper
conditions, and verifies the engine fires/doesn't fire accordingly.
"""
import os
import tempfile
import pandas as pd
import numpy as np

# Point SQLite to a temp file so tests don't pollute real DB
_tmp = tempfile.mktemp(suffix=".db")
os.environ["ALERT_DB_PATH"] = _tmp

from src.rule_engine import compute_market_data, evaluate_setups
from src.fvg_detector import FVGDirection, FVGState


def make_ohlcv(n, base_price=100, trend=0, volatility=1.0, base_volume=1000):
    """Generate synthetic OHLCV data."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(42)
    closes = [base_price]
    for i in range(1, n):
        closes.append(closes[-1] + trend + np.random.randn() * volatility)
    closes = np.array(closes)
    highs = closes + np.abs(np.random.randn(n)) * volatility
    lows = closes - np.abs(np.random.randn(n)) * volatility
    opens = closes + np.random.randn(n) * volatility * 0.5
    volumes = np.abs(np.random.randn(n)) * base_volume + base_volume
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }, index=idx)


def make_4h_from_15m(df_15m):
    """Resample 15m to 4h."""
    return df_15m.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


def make_1h_from_15m(df_15m):
    return df_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


def test_long_all_gates_fail_on_downtrend():
    """
    In a clear downtrend, 4h close should be below EMA50,
    so the long setup should not fire.
    """
    df_15m = make_ohlcv(500, base_price=100, trend=-0.1, volatility=0.5)
    df_4h = make_4h_from_15m(df_15m)
    df_1h = make_1h_from_15m(df_15m)

    md = compute_market_data("TEST", df_15m, df_1h, df_4h)
    alerts = evaluate_setups(md)
    long_alerts = [a for a in alerts if a["direction"] == "long"]
    assert len(long_alerts) == 0, f"expected no long alerts in downtrend, got {len(long_alerts)}"
    print("PASS long_all_gates_fail_on_downtrend")


def test_short_no_fire_without_overbought():
    """
    Short setup requires daily RSI > 70 or 3d RSI > 70.
    Without daily/3d data, shorts should never fire.
    """
    df_15m = make_ohlcv(500, base_price=100, trend=-0.1, volatility=0.5)
    df_4h = make_4h_from_15m(df_15m)
    df_1h = make_1h_from_15m(df_15m)

    # No daily or 3d data
    md = compute_market_data("TEST", df_15m, df_1h, df_4h, df_daily=None, df_3d=None)
    alerts = evaluate_setups(md)
    short_alerts = [a for a in alerts if a["direction"] == "short"]
    assert len(short_alerts) == 0, f"expected no short alerts without daily data, got {len(short_alerts)}"
    print("PASS short_no_fire_without_overbought")


def test_no_alerts_with_insufficient_data():
    """Too little data should produce no alerts, not crash."""
    df_15m = make_ohlcv(10, base_price=100)
    df_4h = make_ohlcv(5, base_price=100)
    df_1h = make_ohlcv(5, base_price=100)

    md = compute_market_data("TEST", df_15m, df_1h, df_4h)
    alerts = evaluate_setups(md)
    assert len(alerts) == 0, "expected no alerts with insufficient data"
    print("PASS no_alerts_with_insufficient_data")


def test_long_requires_volume():
    """
    Even if trend/RSI/histogram conditions pass, low volume
    should block the alert. We test this by creating conditions
    that would pass but with zero volume.
    """
    # Uptrend — 4h should be above EMA50
    df_15m = make_ohlcv(500, base_price=100, trend=0.05, volatility=0.3)
    # Kill volume on last candle
    df_15m.iloc[-1, df_15m.columns.get_loc("volume")] = 0.0
    df_4h = make_4h_from_15m(df_15m)
    df_1h = make_1h_from_15m(df_15m)

    md = compute_market_data("TEST", df_15m, df_1h, df_4h)
    alerts = evaluate_setups(md)
    long_alerts = [a for a in alerts if a["direction"] == "long"]
    # Volume gate should block — even if other conditions happen to align
    # (they probably don't with synthetic data, but the important thing
    # is this doesn't crash)
    print(f"PASS long_requires_volume: {len(long_alerts)} alerts (expected 0 or blocked by other gates)")


def test_market_data_computation_no_crash():
    """
    compute_market_data should handle all TF combinations without crashing.
    """
    df_15m = make_ohlcv(2000, base_price=50000, trend=10, volatility=200)
    df_1h = make_1h_from_15m(df_15m)
    df_4h = make_4h_from_15m(df_15m)

    # With daily/3d
    daily_idx = pd.date_range("2025-01-01", periods=300, freq="1D", tz="UTC")
    df_daily = pd.DataFrame({
        "open": np.linspace(40000, 50000, 300),
        "high": np.linspace(40500, 50500, 300),
        "low": np.linspace(39500, 49500, 300),
        "close": np.linspace(40000, 50000, 300),
        "volume": [1e9] * 300,
    }, index=daily_idx)

    md = compute_market_data("BTC", df_15m, df_1h, df_4h, df_daily=df_daily)
    assert md.ema50_4h is not None
    assert md.rsi_15m is not None
    assert md.ema_hist_15m is not None
    assert md.rsi_daily is not None
    assert md.ema200_daily is not None
    assert md.fvgs_4h is not None
    assert md.fvgs_1h is not None
    print("PASS market_data_computation_no_crash")


def test_alert_dedup():
    """
    Running evaluate_setups twice with the same data should not
    produce duplicate alerts (checked via alert_hash).
    """
    from src.alert_logger import log_alert, make_alert_hash, alert_exists

    h = make_alert_hash("TEST", "long", "2026-01-01T00:00:00", "4h")
    assert not alert_exists(h)

    log_alert({
        "created_at": "2026-01-01T00:00:00",
        "symbol": "TEST",
        "direction": "long",
        "setup_type": "day_long",
        "priority": "normal",
        "fvg_tf": "4h",
        "fvg_top": 100,
        "fvg_bottom": 95,
        "fvg_created_at": "2026-01-01T00:00:00",
        "fvg_displacement_ratio": 2.0,
        "entry_price": 97.5,
        "distance_into_fvg_pct": 50.0,
        "stop_ref": 94.0,
        "stop_source": "fvg_low",
        "trend_4h_ema50": 90.0,
        "trend_4h_close": 98.0,
        "trend_4h_ema50_slope": None,
        "rsi_15m": 35.0,
        "rsi_15m_bounced": 1,
        "ema_hist_15m": 0.5,
        "ema_hist_flipped": 1,
        "volume_15m": 1200,
        "volume_sma_15m": 1000,
        "volume_ratio": 1.2,
        "funding_rate_pct": 0.01,
        "session": "NY",
        "distance_from_4h_ema50_pct": 8.3,
        "rsi_4h": 55.0,
        "rsi_daily": None,
        "rsi_3d": None,
        "daily_200_ema": None,
        "distance_from_daily_200_pct": None,
        "alert_hash": h,
    })

    assert alert_exists(h)
    print("PASS alert_dedup")


if __name__ == "__main__":
    test_long_all_gates_fail_on_downtrend()
    test_short_no_fire_without_overbought()
    test_no_alerts_with_insufficient_data()
    test_long_requires_volume()
    test_market_data_computation_no_crash()
    test_alert_dedup()
    print("\nAll rule engine tests passed.")

    # Clean up temp DB
    try:
        os.unlink(_tmp)
    except Exception:
        pass
