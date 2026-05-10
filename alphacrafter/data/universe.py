"""Universe U: load S&P 500 (or custom) constituent table from CSV."""

from __future__ import annotations

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
