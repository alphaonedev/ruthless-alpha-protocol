"""Minimal, read-only Alpaca Market Data client using the standard library."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta


def _credentials() -> tuple[str, str, str]:
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    base = os.environ.get("APCA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
    if not key or not secret:
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment")
    return key, secret, base


def daily_bars(symbols: list[str], years: int = 3) -> dict[str, list[dict]]:
    """Fetch split/dividend-adjusted daily bars without logging credentials."""
    key, secret, base = _credentials()
    end = date.today()
    start = end - timedelta(days=366 * years + 10)
    query = urllib.parse.urlencode({
        "symbols": ",".join(symbols), "timeframe": "1Day",
        "start": start.isoformat(), "end": end.isoformat(),
        "adjustment": "all", "feed": "iex", "limit": 10000,
    })
    request = urllib.request.Request(
        f"{base}/v2/stocks/bars?{query}",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload.get("bars", {})

