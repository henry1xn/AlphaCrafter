"""Smoke tests for SQLite schema and universe helpers (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alphacrafter.data.universe import load_universe_csv
from alphacrafter.memory import SharedMemory, init_database


class TestSharedMemory(unittest.TestCase):
    def test_init_database_creates_tables(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.db"
            init_database(p)
            sm = SharedMemory(p)
            try:
                sm.connect()
                ver = sm.get_meta("schema_version")
                self.assertEqual(ver, "1")
                cur = sm.connect().execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                names = {r[0] for r in cur.fetchall()}
                self.assertIn("factor_records", names)
                self.assertIn("market_regimes", names)
            finally:
                sm.close()

    def test_record_factor_event(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.db"
            init_database(p)
            sm = SharedMemory(p)
            try:
                rid = sm.record_factor_event("return close.pct_change()", 0.05, 0.4, "effective", in_library=True)
                self.assertGreater(rid, 0)
                rows = sm.list_library_factors()
                self.assertEqual(len(rows), 1)
            finally:
                sm.close()


class TestUniverseCsv(unittest.TestCase):
    def test_load_sample_csv(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "u.csv"
            p.write_text("Symbol,Security\nMSFT,Microsoft\n", encoding="utf-8")
            df = load_universe_csv(p)
            self.assertListEqual(df["ticker"].tolist(), ["MSFT"])


if __name__ == "__main__":
    unittest.main()
