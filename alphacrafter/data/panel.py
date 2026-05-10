"""Build long-format OHLCV panel for U over a date window."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from alphacrafter.data.historical import fetch_daily_ohlcv


def default_date_window(*, trading_days: int = 200, end: date | None = None) -> tuple[date, date]:
    """Approximate calendar span for daily bars (weekends + buffer)."""
    end_d = end or date.today()
    start_d = end_d - timedelta(days=int(trading_days * 1.6) + 10)
    return start_d, end_d


def build_long_panel(
    tickers: list[str],
    *,
    start: date | None = None,
    end: date | None = None,
    trading_days: int = 200,
    sleep_sec: float | None = None,
) -> pd.DataFrame:
    """
    Concatenate per-ticker daily OHLCV into one long DataFrame sorted by date, ticker.
    """
    if end is None or start is None:
        start_d, end_d = default_date_window(trading_days=trading_days, end=end)
        start = start or start_d
        end = end or end_d
    frames: list[pd.DataFrame] = []
    for t in tickers:
        try:
            df = fetch_daily_ohlcv(t, start=start, end=end, sleep_sec=sleep_sec)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def add_forward_return(panel: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Add fwd_ret: next-day close-to-close return within each ticker."""
    if panel.empty:
        return panel
    df = panel.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)["close"]
    df["fwd_ret"] = g.pct_change(horizon).shift(-horizon)
    return df
