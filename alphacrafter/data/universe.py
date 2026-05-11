"""Universe U: load S&P 500 (or custom) constituent table from CSV."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from alphacrafter.config.settings import PROJECT_ROOT, RAW_DIR

_SYMBOL_COLUMNS = ("Symbol", "symbol", "Ticker", "ticker", "SYM", "sym")


def default_universe_csv() -> Path:
    """Default path after running scripts/scrape_sp500.py."""
    return RAW_DIR / "sp500_wiki.csv"


def load_universe_csv(path: str | Path | None = None) -> pd.DataFrame:
    """
    Read universe CSV; normalize a `ticker` column for downstream fetches.
    Wikipedia export uses 'Symbol' and 'Security', etc.
    """
    p = Path(path) if path else default_universe_csv()
    if not p.is_file():
        raise FileNotFoundError(
            f"Universe CSV not found: {p}. Run: python scripts/scrape_sp500.py --out {p}"
        )
    df = pd.read_csv(p)
    sym_col = next((c for c in _SYMBOL_COLUMNS if c in df.columns), None)
    if sym_col is None:
        raise ValueError(
            f"No symbol column in {p}; expected one of {_SYMBOL_COLUMNS}. Columns: {list(df.columns)}"
        )
    out = df.copy()
    out["ticker"] = out[sym_col].astype(str).str.strip()
    return out


def project_relative(path: str | Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    pp = Path(path)
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)


def load_crypto_universe(
    data_dir: str | Path,
    *,
    ticker_limit: int | None = None,
    rank_by: str = "volume",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Build universe **U** from filenames in ``data_dir`` (``*.csv`` / ``*.parquet``).

    ``rank_by``: ``volume`` (default, liquidity proxy) or ``none`` (filename order).
    ``ticker_limit``: keep top-N after ranking; ``None`` means all symbols.
    """
    from alphacrafter.config.settings import ORCH_TICKER_LIMIT
    from alphacrafter.data.local_klines import list_kline_files, rank_crypto_symbols_by_volume, symbol_from_kline_path

    rb = (rank_by or "volume").strip().lower()
    if rb in {"none", "file", "name"}:
        paths = list_kline_files(data_dir)
        rows = [{"ticker": symbol_from_kline_path(p), "volume_score": 0.0} for p in paths]
        ranked = pd.DataFrame(rows)
    elif rb in {"volume", "vol"}:
        lb = int(lookback_days) if lookback_days is not None else int(os.getenv("ALPHACRAFTER_CRYPTO_LOOKBACK_DAYS", "90") or "90")
        ranked = rank_crypto_symbols_by_volume(data_dir, lookback_days=lb)
    else:
        raise ValueError("rank_by must be 'volume' or 'none' (market cap not available from OHLCV files).")
    if ranked.empty:
        raise FileNotFoundError(
            f"No kline files (*.csv / *.parquet) found under: {Path(data_dir).resolve()}"
        )
    lim = int(ticker_limit if ticker_limit is not None else ORCH_TICKER_LIMIT)
    out = ranked.head(lim)[["ticker"]].copy()
    return out.reset_index(drop=True)
