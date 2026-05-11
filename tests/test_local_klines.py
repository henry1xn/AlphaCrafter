"""Local crypto k-line reader (no network)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from alphacrafter.data.local_klines import (
    list_kline_files,
    load_crypto_long_panel,
    read_single_kline_file,
    symbol_from_kline_path,
)
from alphacrafter.data.universe import load_crypto_universe


class TestLocalKlines(unittest.TestCase):
    def test_symbol_from_path(self) -> None:
        self.assertEqual(symbol_from_kline_path(Path("btc-usdt.csv")), "BTCUSDT")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nested = root / "ETHUSDT" / "chunk.parquet"
            nested.parent.mkdir(parents=True)
            self.assertEqual(symbol_from_kline_path(nested, data_root=root), "ETHUSDT")

    def test_list_kline_files_nested_symbol_dirs(self) -> None:
        """Binance-style layout: ROOT/SYMBOL/*.parquet (and deeper)."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f1 = root / "BTCUSDT" / "chunk.parquet"
            f1.parent.mkdir(parents=True)
            f1.touch()
            f2 = root / "ETHUSDT" / "monthly" / "m.parquet"
            f2.parent.mkdir(parents=True)
            f2.touch()
            found = list_kline_files(root)
            self.assertEqual({p.resolve() for p in found}, {f1.resolve(), f2.resolve()})

    def test_read_csv_and_panel(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            df = pd.DataFrame(
                {
                    "open_time": pd.date_range("2024-01-01", periods=5, freq="8h"),
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "volume": 100.0,
                }
            )
            df.to_csv(root / "AAAUSDT.csv", index=False)
            out = read_single_kline_file(root / "AAAUSDT.csv")
            self.assertEqual(out["ticker"].iloc[0], "AAAUSDT")
            self.assertGreaterEqual(len(out), 1)

            u = load_crypto_universe(root, ticker_limit=10, rank_by="none")
            self.assertListEqual(u["ticker"].tolist(), ["AAAUSDT"])

            panel = load_crypto_long_panel(
                ["AAAUSDT"],
                root,
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                trading_days=30,
            )
            self.assertFalse(panel.empty)


if __name__ == "__main__":
    unittest.main()
