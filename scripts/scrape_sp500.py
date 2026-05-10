"""
Fetch current S&P 500 constituents from Wikipedia and save as CSV (universe U).

Usage:
  python scripts/scrape_sp500.py --out data/raw/sp500_wiki.csv
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_UA = (
    "AlphaCrafter/1.0 (+https://github.com/local/AlphaCrafter; research data pipeline)"
)


def fetch_sp500_table(url: str = WIKI_URL, timeout: float = 60.0) -> pd.DataFrame:
    headers = {"User-Agent": DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    buf = io.StringIO(resp.text)
    tables = pd.read_html(buf)
    if not tables:
        raise RuntimeError("No HTML tables parsed from Wikipedia response.")
    # First table is the S&P 500 list on the standard page layout.
    df = tables[0].copy()
    # Normalize column names for downstream code
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape S&P 500 tickers from Wikipedia.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/sp500_wiki.csv"),
        help="Output CSV path (default: data/raw/sp500_wiki.csv)",
    )
    parser.add_argument("--url", type=str, default=WIKI_URL, help="Override Wikipedia URL.")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_sp500_table(url=args.url)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
