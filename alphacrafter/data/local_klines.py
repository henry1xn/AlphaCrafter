"""Load cryptocurrency OHLCV from local CSV / Parquet (no Yahoo / no wiki universe)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_KLINE_SUFFIXES = (".csv", ".parquet", ".pq")


def symbol_from_kline_path(path: Path) -> str:
    """BTCUSDT.csv -> BTCUSDT; ETH-USDT.parquet -> ETHUSDT."""
    stem = path.stem.upper().strip()
    return stem.replace("-", "").replace("_", "")


def list_kline_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Crypto data directory not found: {root}")
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in _KLINE_SUFFIXES:
            out.append(p)
    return out


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {path}")


def _find_datetime_column(df: pd.DataFrame) -> str | None:
    cols = {c.lower().strip(): c for c in df.columns}
    for key in (
        "open_time",
        "timestamp",
        "time",
        "datetime",
        "date",
        "ts",
        "close_time",
    ):
        if key in cols:
            return cols[key]
    return None


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
    elif pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        mx = float(v.max()) if len(v) else 0.0
        if mx > 1e15:
            out = pd.to_datetime(v, unit="us", errors="coerce", utc=True)
        elif mx > 1e12:
            out = pd.to_datetime(v, unit="ms", errors="coerce", utc=True)
        else:
            out = pd.to_datetime(v, unit="s", errors="coerce", utc=True)
    else:
        out = pd.to_datetime(s, errors="coerce", utc=True)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    m = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in m:
                return m[n]
        return None

    dt_col = _find_datetime_column(df)
    if dt_col is None:
        raise ValueError(f"No datetime column found. Columns: {list(df.columns)}")

    o = pick("open", "o")
    h = pick("high", "h")
    l = pick("low", "l")
    c = pick("close", "c")
    v = pick("volume", "vol", "v", "quote_volume", "quote asset volume")
    if not all([o, h, l, c]):
        raise ValueError(f"Missing OHLC columns after mapping. Columns: {list(df.columns)}")
    if v is None:
        df = df.copy()
        df["_vol0"] = 0.0
        v = "_vol0"

    out = pd.DataFrame(
        {
            "date": _parse_datetime_series(df[dt_col]),
            "open": pd.to_numeric(df[o], errors="coerce"),
            "high": pd.to_numeric(df[h], errors="coerce"),
            "low": pd.to_numeric(df[l], errors="coerce"),
            "close": pd.to_numeric(df[c], errors="coerce"),
            "volume": pd.to_numeric(df[v], errors="coerce").fillna(0.0),
        }
    )
    return out.dropna(subset=["date", "close"])


def _aggregate_to_daily_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday rows to one OHLCV row per calendar day (24/7 markets)."""
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    day = df["date"].dt.normalize()
    if not day.duplicated().any():
        return df.assign(date=day)
    tmp = df.copy()
    tmp["_day"] = day
    agg = (
        tmp.groupby("_day", sort=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"_day": "date"})
    )
    return agg


def read_single_kline_file(path: Path, ticker: str | None = None) -> pd.DataFrame:
    """Parse one file into long-format columns: date, open, high, low, close, volume, ticker."""
    raw = _read_any(path)
    base = _rename_ohlcv(raw)
    base = _aggregate_to_daily_if_needed(base)
    sym = ticker or symbol_from_kline_path(path)
    base["ticker"] = sym
    base = base.sort_values("date").reset_index(drop=True)
    return base[["date", "ticker", "open", "high", "low", "close", "volume"]]


def volume_score_for_symbol(path: Path, *, lookback_days: int = 90) -> tuple[str, float]:
    """Approximate liquidity: sum(volume) over last ``lookback_days`` calendar days in file."""
    tkr = symbol_from_kline_path(path)
    try:
        df = read_single_kline_file(path, ticker=tkr)
    except Exception:
        return tkr, 0.0
    if df.empty:
        return tkr, 0.0
    end = df["date"].max()
    if pd.isna(end):
        return tkr, 0.0
    start = end - pd.Timedelta(days=int(lookback_days))
    sub = df.loc[df["date"] >= start]
    return tkr, float(sub["volume"].sum())


def rank_crypto_symbols_by_volume(
    data_dir: str | Path,
    *,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Discover all symbols from filenames; rank by recent total volume (proxy for liquidity).

    Market-cap is not available from OHLCV files — use ``ALPHACRAFTER_CRYPTO_RANK_BY=volume`` only.
    """
    rows: list[dict[str, object]] = []
    for path in list_kline_files(data_dir):
        tkr, score = volume_score_for_symbol(path, lookback_days=lookback_days)
        rows.append({"ticker": tkr, "volume_score": score, "path": str(path)})
    if not rows:
        return pd.DataFrame(columns=["ticker", "volume_score", "path"])
    out = pd.DataFrame(rows).sort_values("volume_score", ascending=False).reset_index(drop=True)
    return out


def load_crypto_long_panel(
    tickers: list[str],
    data_dir: str | Path,
    *,
    start: date | None = None,
    end: date | None = None,
    trading_days: int = 200,
) -> pd.DataFrame:
    """
    Concatenate local klines for ``tickers`` into one long panel (24/7 calendar, no Yahoo).

    If ``start``/``end`` omitted, uses last ``trading_days`` **calendar** days ending today
    (bars-per-day varies for crypto — this is an approximate window).
    """
    root = Path(data_dir).expanduser().resolve()
    end_d = end or date.today()
    start_d = start or (end_d - timedelta(days=max(int(trading_days), 1)))

    want = {t.upper().strip().replace("-", "").replace("_", "") for t in tickers}
    frames: list[pd.DataFrame] = []
    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for path in list_kline_files(root):
        sym = symbol_from_kline_path(path)
        if sym not in want:
            continue
        try:
            df = read_single_kline_file(path, ticker=sym)
        except Exception:
            continue
        if df.empty:
            continue
        d = df["date"]
        sub = df.loc[(d >= lo) & (d <= hi)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
