"""Vectorized long/short cross-sectional backtest on daily returns."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _bars_per_year() -> float:
    """252 ~ equities; 365 ~ 24/7 daily crypto bars (set ALPHACRAFTER_BARS_PER_YEAR)."""
    raw = os.getenv("ALPHACRAFTER_BARS_PER_YEAR", "").strip()
    if raw:
        return float(raw)
    if os.getenv("ALPHACRAFTER_ASSET_CLASS", "").strip().lower() == "crypto":
        return 365.0
    return 252.0


def cross_sectional_zscore(signals: pd.DataFrame) -> pd.DataFrame:
    mu = signals.mean(axis=1)
    sd = signals.std(axis=1).replace(0, np.nan)
    z = signals.sub(mu, axis=0).div(sd, axis=0)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def daily_portfolio_metrics(port: pd.Series) -> dict[str, float]:
    """
    Sharpe / CAGR / MDD from a **daily portfolio return** series (already aligned to holding period).
    """
    port = pd.Series(port, dtype=float).dropna()
    if port.empty:
        return {
            "sharpe_ann": 0.0,
            "mean_daily": 0.0,
            "cum_return": 0.0,
            "n": 0.0,
            "ann_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    bpy = _bars_per_year()
    mu = float(port.mean())
    sd = float(port.std(ddof=1))
    sharpe = (mu / sd * np.sqrt(bpy)) if sd > 1e-12 else 0.0
    cum = float((1.0 + port).prod() - 1.0)
    n = int(len(port))
    equity = (1.0 + port).cumprod()
    end_eq = float(equity.iloc[-1])
    ann_ret_pct = (end_eq ** (bpy / max(n, 1)) - 1.0) * 100.0 if n > 0 else 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    mdd_pct = float(dd.min()) * 100.0
    return {
        "sharpe_ann": float(sharpe),
        "mean_daily": mu,
        "cum_return": cum,
        "n": float(n),
        "ann_return_pct": float(ann_ret_pct),
        "max_drawdown_pct": float(mdd_pct),
    }


def backtest_long_short(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    signal_lag: int = 1,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Dollar-neutral cross-sectional portfolio.

    ``signal_lag=1`` uses signal known at t-1 against return from t-1 to t (``returns`` rows
    should be close-to-close pct_change indexed at end date t).
    """
    sig = signals.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    z = cross_sectional_zscore(sig)
    if signal_lag:
        z = z.shift(signal_lag)
    fwd = returns.reindex_like(z).fillna(0.0)
    denom = z.abs().sum(axis=1).replace(0, np.nan)
    port = (z * fwd).sum(axis=1) / denom
    port = port.dropna()
    metrics = daily_portfolio_metrics(port)
    return port, metrics


def pivot_close_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide daily returns from long OHLCV panel."""
    if panel.empty:
        return pd.DataFrame()
    wide = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide.pct_change().iloc[1:]


def pivot_signal_from_long(panel: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    """Align long factor series with panel rows into wide matrix."""
    df = panel[["date", "ticker"]].copy()
    df["f"] = factor.values
    return df.pivot(index="date", columns="ticker", values="f").sort_index()
