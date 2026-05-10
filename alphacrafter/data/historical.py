"""Daily OHLCV via Yahoo Finance chart API (requests + JSON), no extra SDK."""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from alphacrafter.config.settings import HTTP_SLEEP_SEC, PROCESSED_DIR

_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; AlphaCrafter/1.0; +https://example.invalid; research data pipeline)"
)


def to_yahoo_symbol(ticker: str) -> str:
    """Map 'BRK.B' to Yahoo-style 'BRK-B'."""
    return ticker.strip().upper().replace(".", "-")


def _fmt_period(ts: date | datetime | str) -> int:
    if isinstance(ts, str):
        t = pd.Timestamp(ts)
    elif isinstance(ts, datetime):
        t = pd.Timestamp(ts)
    else:
        t = pd.Timestamp(ts)
    return int(t.timestamp())


def _parse_chart_json(payload: dict[str, Any], source_ticker: str) -> pd.DataFrame:
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()
    res = results[0]
    ts = res.get("timestamp") or []
    if not ts:
        return pd.DataFrame()
    quotes = (res.get("indicators") or {}).get("quote") or [{}]
    q = quotes[0] if quotes else {}
    ny = ZoneInfo("America/New_York")
    dates = (
        pd.to_datetime(pd.Series(ts, dtype="int64"), unit="s", utc=True)
        .dt.tz_convert(ny)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    df = pd.DataFrame(
        {
            "date": dates,
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        }
    )
    df = df.dropna(subset=["close"])
    df["ticker"] = source_ticker.strip().upper()
    return df.sort_values("date").reset_index(drop=True)


def fetch_daily_ohlcv(
    ticker: str,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    timeout: float = 60.0,
    sleep_sec: float | None = None,
) -> pd.DataFrame:
    """
    Download daily bars from Yahoo's public chart endpoint.

    If both ``start`` and ``end`` are omitted, uses ``range=max`` for full history
    (subject to Yahoo limits).
    """
    sym = to_yahoo_symbol(ticker)
    url = _YAHOO_CHART.format(symbol=sym)
    headers = {"User-Agent": _DEFAULT_UA}

    delay = HTTP_SLEEP_SEC if sleep_sec is None else sleep_sec
    time.sleep(max(delay, 0.0))

    if start is not None and end is not None:
        params: dict[str, str | int] = {
            "interval": "1d",
            "period1": _fmt_period(start),
            "period2": _fmt_period(end),
        }
    else:
        params = {"interval": "1d", "range": "max"}

    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return _parse_chart_json(resp.json(), source_ticker=ticker)


def fetch_daily_panel(
    tickers: list[str],
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    on_error: Literal["skip", "raise"] = "skip",
) -> pd.DataFrame:
    """Sequential download; returns long-format concatenated frame."""
    frames: list[pd.DataFrame] = []
    for t in tickers:
        try:
            df = fetch_daily_ohlcv(t, start=start, end=end)
            if not df.empty:
                frames.append(df)
        except Exception:
            if on_error == "raise":
                raise
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_panel_csv(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Persist long-format OHLCV panel to processed/ for reuse."""
    if df.empty:
        raise ValueError("Cannot save empty DataFrame.")
    out = Path(path) if path else (PROCESSED_DIR / "ohlcv_panel.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
