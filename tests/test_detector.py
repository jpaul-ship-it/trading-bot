"""
Offline validation of the FVG detector using synthetic OHLCV.
Builds specific scenarios and asserts the detector classifies them correctly.
"""
import pandas as pd
import numpy as np
from src.fvg_detector import detect_fvgs, FVGDirection, FVGState


def make_df(candles):
    """candles: list of (o, h, l, c, v). Index is sequential minutes."""
    idx = pd.date_range("2026-01-01", periods=len(candles), freq="15min", tz="UTC")
    df = pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"], index=idx)
    return df


def test_bullish_unfilled():
    # 25 baseline candles to warm up ATR, then a clean bullish FVG, then quiet candles above
    base = [(100, 101, 99, 100, 1000)] * 25
    # displacement: candle i-1 high=101, candle i low=105 high=120 (big range), candle i+1 low=115
    fvg = [
        (101, 102, 100, 101, 1000),   # i-1
        (102, 120, 102, 118, 5000),   # i   displacement (range=18, ATR~2 so huge)
        (118, 119, 115, 117, 1000),   # i+1 low=115 > i-1 high=102 -> bullish FVG [102, 115]
    ]
    # price stays above the gap — never tags
    after = [(117, 118, 116, 117, 1000)] * 5
    df = make_df(base + fvg + after)

    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    assert len(fvgs) == 1, f"expected 1 FVG, got {len(fvgs)}"
    f = fvgs[0]
    assert f.direction == FVGDirection.BULLISH
    assert f.state == FVGState.UNFILLED, f"expected unfilled, got {f.state}"
    assert abs(f.bottom - 102) < 0.01
    assert abs(f.top - 115) < 0.01
    print(f"PASS bullish_unfilled: {f.state.value}, gap=[{f.bottom}, {f.top}]")


def test_bullish_tagged():
    base = [(100, 101, 99, 100, 1000)] * 25
    fvg = [
        (101, 102, 100, 101, 1000),
        (102, 120, 102, 118, 5000),
        (118, 119, 115, 117, 1000),
    ]
    # price wicks into gap but closes above top
    after = [
        (117, 118, 116, 117, 1000),
        (117, 117, 110, 116, 1000),   # wick into gap (low=110 is inside [102,115]), close=116 > top=115
        (116, 117, 115, 116, 1000),
    ]
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    f = fvgs[0]
    assert f.state == FVGState.TAGGED, f"expected tagged, got {f.state}"
    assert f.first_tag_at is not None
    print(f"PASS bullish_tagged: {f.state.value}, first_tag_at={f.first_tag_at}")


def test_bullish_filled():
    base = [(100, 101, 99, 100, 1000)] * 25
    fvg = [
        (101, 102, 100, 101, 1000),
        (102, 120, 102, 118, 5000),
        (118, 119, 115, 117, 1000),
    ]
    # candle closes below FVG top (115) but above bottom (102) -> filled
    after = [
        (117, 118, 110, 112, 1000),   # close=112 is below top=115, above bottom=102
    ]
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    f = fvgs[0]
    assert f.state == FVGState.FILLED, f"expected filled, got {f.state}"
    print(f"PASS bullish_filled: {f.state.value}, filled_at={f.filled_at}")


def test_bullish_invalidated():
    base = [(100, 101, 99, 100, 1000)] * 25
    fvg = [
        (101, 102, 100, 101, 1000),
        (102, 120, 102, 118, 5000),
        (118, 119, 115, 117, 1000),
    ]
    # close below FVG bottom (102) -> invalidated
    after = [
        (117, 118, 95, 98, 1000),
    ]
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    f = fvgs[0]
    assert f.state == FVGState.INVALIDATED, f"expected invalidated, got {f.state}"
    print(f"PASS bullish_invalidated: {f.state.value}")


def test_bearish_unfilled():
    base = [(100, 101, 99, 100, 1000)] * 25
    # bearish FVG: candle[i-1].low > candle[i+1].high
    fvg = [
        (100, 101, 99, 100, 1000),     # i-1 low=99
        (99, 99, 82, 84, 5000),        # i   big range
        (84, 85, 82, 83, 1000),        # i+1 high=85 < i-1 low=99 -> bearish FVG [85, 99]
    ]
    after = [(83, 84, 82, 83, 1000)] * 5
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    assert len(fvgs) == 1
    f = fvgs[0]
    assert f.direction == FVGDirection.BEARISH
    assert f.state == FVGState.UNFILLED
    assert abs(f.bottom - 85) < 0.01
    assert abs(f.top - 99) < 0.01
    print(f"PASS bearish_unfilled: {f.state.value}, gap=[{f.bottom}, {f.top}]")


def test_bearish_filled():
    base = [(100, 101, 99, 100, 1000)] * 25
    fvg = [
        (100, 101, 99, 100, 1000),
        (99, 99, 82, 84, 5000),
        (84, 85, 82, 83, 1000),
    ]
    # close above bottom (85) but below top (99) -> filled
    after = [(83, 92, 82, 90, 1000)]
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    f = fvgs[0]
    assert f.state == FVGState.FILLED, f"expected filled, got {f.state}"
    print(f"PASS bearish_filled: {f.state.value}")


def test_stale():
    base = [(100, 101, 99, 100, 1000)] * 25
    fvg = [
        (101, 102, 100, 101, 1000),
        (102, 120, 102, 118, 5000),
        (118, 119, 115, 117, 1000),
    ]
    # 60 quiet candles above the gap — never tag, should go stale (stale_after=50)
    after = [(117, 118, 116, 117, 1000)] * 60
    df = make_df(base + fvg + after)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    f = fvgs[0]
    assert f.state == FVGState.STALE, f"expected stale, got {f.state}"
    print(f"PASS stale: {f.state.value}, age={f.candles_since_creation}")


def test_displacement_filter_rejects_small_candles():
    # Tiny candles with a "gap" but no real displacement
    base = [(100, 101, 99, 100, 1000)] * 25
    weak = [
        (100.0, 100.2, 99.9, 100.1, 1000),
        (100.1, 100.3, 100.0, 100.2, 1000),   # range=0.3, tiny — below 1.5x ATR
        (100.2, 100.5, 100.4, 100.4, 1000),   # technically low=100.4 > prev-prev high=100.2
    ]
    df = make_df(base + weak)
    fvgs = detect_fvgs(df, "TEST", "15m", atr_period=20, displacement_mult=1.5, stale_after=50)
    assert len(fvgs) == 0, f"expected no FVGs (displacement filter), got {len(fvgs)}"
    print(f"PASS displacement_filter_rejects: 0 FVGs as expected")


if __name__ == "__main__":
    test_bullish_unfilled()
    test_bullish_tagged()
    test_bullish_filled()
    test_bullish_invalidated()
    test_bearish_unfilled()
    test_bearish_filled()
    test_stale()
    test_displacement_filter_rejects_small_candles()
    print("\nAll detector tests passed.")
