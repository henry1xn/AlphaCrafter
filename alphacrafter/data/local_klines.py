"""Load cryptocurrency OHLCV from local CSV / Parquet (no Yahoo / no wiki universe)."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_KLINE_SUFFIXES = (".csv", ".parquet", ".pq")


def _kline_file_ok(p: Path) -> bool:
    """True if path looks like a k-line data file (not only strict .suffix — handles odd FS / names)."""
    if not p.is_file():
        return False
    s = p.suffix.lower()
    if s in _KLINE_SUFFIXES:
        return True
    n = p.name.lower()
    return n.endswith(".parquet.gz") or n.endswith(".csv.gz")


def symbol_from_kline_path(path: Path, *, data_root: Path | None = None) -> str:
    """
    Resolve ticker: prefer parent folder under ``data_root`` (e.g. .../klines/BTCUSDT/monthly.parquet).

    Otherwise infer from filename stem (e.g. ``BTCUSDT-1d-2024-01`` → ``BTCUSDT``).
    """
    if data_root is not None:
        try:
            rel = path.resolve().relative_to(data_root.resolve())
            parts = rel.parts
            if len(parts) >= 2:
                sym_dir = parts[0].upper().replace("-", "").replace("_", "")
                if sym_dir and len(sym_dir) >= 4:
                    return sym_dir
        except ValueError:
            pass
    stem = path.stem.upper().strip()
    # Binance-style flat name: BTCUSDT-1d-2024-01
    for sep in ("-", "_"):
        if sep in stem:
            head = stem.split(sep)[0]
            if len(head) >= 6 and (
                head.endswith("USDT")
                or head.endswith("USDC")
                or head.endswith("BUSD")
                or head.endswith("USD")
            ):
                return head.replace("-", "").replace("_", "")
    return stem.replace("-", "").replace("_", "")


def list_kline_files(data_dir: str | Path) -> list[Path]:
    """
    Collect k-line files under ``data_dir``.

    - **Flat**: ``ROOT/BTCUSDT.parquet``
    - **Binance-style** (常见): ``ROOT/ADAUSDT/*.parquet`` — 对每个一级子目录做 ``rglob``，
      避免在部分 NFS 上 ``ROOT.glob('**/*.parquet')`` 不返回结果的问题。

    关闭子目录扫描：``ALPHACRAFTER_CRYPTO_SCAN_SUBDIRS=0``（仅根目录下的文件）。
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Crypto data directory not found: {root}")
    seen: set[str] = set()
    out: list[Path] = []

    def add(p: Path) -> None:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)

    scan_sub = os.getenv("ALPHACRAFTER_CRYPTO_SCAN_SUBDIRS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

    for p in sorted(root.iterdir()):
        if p.is_file() and _kline_file_ok(p):
            add(p)
        elif scan_sub and p.is_dir():
            try:
                for q in p.rglob("*"):
                    if _kline_file_ok(q):
                        add(q)
            except OSError:
                continue

    if scan_sub and not out:
        for q in root.rglob("*"):
            if _kline_file_ok(q):
                add(q)

    return sorted(out, key=lambda x: str(x))


def _read_any(path: Path) -> pd.DataFrame:
    n = path.name.lower()
    suf = path.suffix.lower()
    if suf == ".csv" or n.endswith(".csv.gz"):
        return pd.read_csv(path, compression="infer")
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if n.endswith(".parquet.gz") or n.endswith(".pq.gz"):
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


def read_single_kline_file(
    path: Path,
    ticker: str | None = None,
    *,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Parse one file into long-format columns: date, open, high, low, close, volume, ticker."""
    raw = _read_any(path)
    base = _rename_ohlcv(raw)
    base = _aggregate_to_daily_if_needed(base)
    sym = ticker or symbol_from_kline_path(path, data_root=data_root)
    base["ticker"] = sym
    base = base.sort_values("date").reset_index(drop=True)
    return base[["date", "ticker", "open", "high", "low", "close", "volume"]]


def volume_score_for_symbol(
    path: Path,
    *,
    data_root: Path,
    lookback_days: int = 90,
) -> tuple[str, float]:
    """Approximate liquidity: sum(volume) over last ``lookback_days`` calendar days in file."""
    tkr = symbol_from_kline_path(path, data_root=data_root)
    try:
        df = read_single_kline_file(path, ticker=tkr, data_root=data_root)
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
    Discover symbols; rank by recent total volume (summed across all files per ticker).
    """
    root = Path(data_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for path in list_kline_files(root):
        tkr, score = volume_score_for_symbol(path, data_root=root, lookback_days=lookback_days)
        rows.append({"ticker": tkr, "volume_score": score})
    if not rows:
        return pd.DataFrame(columns=["ticker", "volume_score"])
    df = pd.DataFrame(rows)
    out = df.groupby("ticker", as_index=False)["volume_score"].sum()
    return out.sort_values("volume_score", ascending=False).reset_index(drop=True)


def _dedupe_panel_dates(out: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate (ticker, date) from multiple monthly files."""
    if out.empty or not out.duplicated(subset=["ticker", "date"]).any():
        return out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return (
        out.groupby(["ticker", "date"], sort=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


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

    If ``start``/``end`` omitted, uses last ``trading_days`` **calendar** days ending today.
    """
    root = Path(data_dir).expanduser().resolve()
    end_d = end or date.today()
    start_d = start or (end_d - timedelta(days=max(int(trading_days), 1)))

    want = {t.upper().strip().replace("-", "").replace("_", "") for t in tickers}
    frames: list[pd.DataFrame] = []
    lo = pd.Timestamp(start_d)
    hi = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for path in list_kline_files(root):
        sym = symbol_from_kline_path(path, data_root=root)
        if sym not in want:
            continue
        try:
            df = read_single_kline_file(path, ticker=sym, data_root=root)
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
    return _dedupe_panel_dates(out)
