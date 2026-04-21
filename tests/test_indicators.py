"""
Tests for the indicators module.
"""
import pandas as pd
import numpy as np
from src.indicators import (
    rsi, ema, ema_histogram, volume_sma,
    ema_slope_flat_or_down,
    rsi_bounced_above, rsi_bounced_below,
    histogram_flipped_positive, histogram_flipped_negative,
    get_session,
)


def test_rsi_range():
    """RSI should always be between 0 and 100."""
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(200)) + 100)
    r = rsi(prices, 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), "RSI out of range"
    print("PASS rsi_range")


def test_rsi_overbought():
    """Steadily rising prices should push RSI > 70."""
    prices = pd.Series([100 + i * 2 for i in range(50)])
    r = rsi(prices, 14)
    assert r.iloc[-1] > 70, f"expected RSI > 70 for rising prices, got {r.iloc[-1]:.1f}"
    print(f"PASS rsi_overbought: RSI={r.iloc[-1]:.1f}")


def test_rsi_oversold():
    """Steadily falling prices should push RSI < 30."""
    prices = pd.Series([200 - i * 2 for i in range(50)])
    r = rsi(prices, 14)
    assert r.iloc[-1] < 30, f"expected RSI < 30 for falling prices, got {r.iloc[-1]:.1f}"
    print(f"PASS rsi_oversold: RSI={r.iloc[-1]:.1f}")


def test_ema_tracks_price():
    """EMA should converge toward a constant price."""
    prices = pd.Series([100.0] * 100)
    e = ema(prices, 20)
    assert abs(e.iloc[-1] - 100.0) < 0.01, f"EMA should be ~100, got {e.iloc[-1]}"
    print("PASS ema_tracks_price")


def test_volume_sma():
    """Volume SMA of constant volume should equal that volume."""
    idx = pd.date_range("2026-01-01", periods=50, freq="15min")
    df = pd.DataFrame({"volume": [1000.0] * 50}, index=idx)
    vs = volume_sma(df, 20)
    assert abs(vs.iloc[-1] - 1000.0) < 0.01
    print("PASS volume_sma")


def test_histogram_flip_positive():
    """Detect neg→pos flip."""
    # EMA histogram: negative then positive
    hist = pd.Series([-5, -3, -1, 0.5, 2])
    assert histogram_flipped_positive(hist, lookback=3)
    print("PASS histogram_flip_positive")


def test_histogram_flip_negative():
    """Detect pos→neg flip."""
    hist = pd.Series([5, 3, 1, -0.5, -2])
    assert histogram_flipped_negative(hist, lookback=3)
    print("PASS histogram_flip_negative")


def test_histogram_no_false_flip():
    """No flip when histogram stays positive."""
    hist = pd.Series([1, 2, 3, 4, 5])
    assert not histogram_flipped_positive(hist, lookback=3)
    assert not histogram_flipped_negative(hist, lookback=3)
    print("PASS histogram_no_false_flip")


def test_rsi_bounce_above():
    """RSI dips below 30 then recovers."""
    r = pd.Series([40, 35, 28, 25, 29, 35])
    assert rsi_bounced_above(r, 30, lookback=5)
    print("PASS rsi_bounce_above")


def test_rsi_bounce_above_no_dip():
    """No bounce if RSI never went below 30."""
    r = pd.Series([50, 45, 40, 35, 32, 38])
    assert not rsi_bounced_above(r, 30, lookback=5)
    print("PASS rsi_bounce_above_no_dip")


def test_rsi_bounce_below():
    """RSI spikes above 70 then falls back."""
    r = pd.Series([60, 65, 72, 75, 71, 65])
    assert rsi_bounced_below(r, 70, lookback=5)
    print("PASS rsi_bounce_below")


def test_ema_slope_down():
    """Declining EMA should be detected as down."""
    vals = pd.Series([100 - i * 0.5 for i in range(20)])
    assert ema_slope_flat_or_down(vals, lookback=10)
    print("PASS ema_slope_down")


def test_ema_slope_up():
    """Rising EMA should not pass the flat/down check."""
    vals = pd.Series([100 + i * 0.5 for i in range(20)])
    assert not ema_slope_flat_or_down(vals, lookback=10)
    print("PASS ema_slope_up")


def test_session_detection():
    """Verify session boundaries."""
    assert get_session(pd.Timestamp("2026-01-01 03:00", tz="UTC")) == "Asia"
    assert get_session(pd.Timestamp("2026-01-01 10:00", tz="UTC")) == "London"
    assert get_session(pd.Timestamp("2026-01-01 15:00", tz="UTC")) == "NY"
    assert get_session(pd.Timestamp("2026-01-01 22:00", tz="UTC")) == "Off"
    print("PASS session_detection")


if __name__ == "__main__":
    test_rsi_range()
    test_rsi_overbought()
    test_rsi_oversold()
    test_ema_tracks_price()
    test_volume_sma()
    test_histogram_flip_positive()
    test_histogram_flip_negative()
    test_histogram_no_false_flip()
    test_rsi_bounce_above()
    test_rsi_bounce_above_no_dip()
    test_rsi_bounce_below()
    test_ema_slope_down()
    test_ema_slope_up()
    test_session_detection()
    print("\nAll indicator tests passed.")
