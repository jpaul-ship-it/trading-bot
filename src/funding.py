"""
Funding rate fetcher.

Pulls the latest funding rate from Binance Futures public API.
No API key required for this endpoint.

Spec references:
  - Long G: log funding rate
  - Short G: gate on funding rate <= +0.005%
"""

from __future__ import annotations

from typing import Optional
import requests

BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Map our symbols to Binance perps tickers
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
}


def fetch_funding_rate(symbol: str) -> Optional[float]:
    """
    Fetch the most recent funding rate for a symbol.
    Returns the rate as a float (e.g. 0.0001 = 0.01%).
    Returns None if fetch fails.
    """
    binance_sym = BINANCE_SYMBOLS.get(symbol)
    if not binance_sym:
        return None

    try:
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": binance_sym, "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and len(data) > 0:
            return float(data[0]["fundingRate"])
    except Exception:
        pass
    return None


def funding_rate_pct(symbol: str) -> Optional[float]:
    """
    Returns funding rate as a percentage (e.g. 0.01 means 0.01%).
    This matches the spec's notation: "funding rate <= +0.005%"
    """
    rate = fetch_funding_rate(symbol)
    if rate is not None:
        return rate * 100
    return None
