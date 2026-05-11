"""Paper Table 1 chronological splits (dataset phases)."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from alphacrafter.data.splits import (
    EVAL_SPLITS,
    EXPECTED_TRADING_DAYS_SP500,
    count_unique_trading_dates,
    filter_panel_to_date_range,
    normalize_split_name,
    paper_split_range,
    split_metadata,
    yahoo_period2_exclusive_end,
)


class TestPaperSplits(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_split_name("train"), "training")
        self.assertEqual(normalize_split_name("val"), "validation")
        self.assertEqual(normalize_split_name("bt"), "backtesting")
        self.assertEqual(normalize_split_name("live"), "live_trading")

    def test_ranges_monotonic(self) -> None:
        phases = ["training", "validation", "backtesting", "live_trading"]
        ends: list[date] = []
        for p in phases:
            s, e = paper_split_range(p)
            self.assertLessEqual(s, e)
            ends.append(e)
        for a, b in zip(ends, ends[1:]):
            self.assertLess(a, b)

    def test_backtesting_window(self) -> None:
        s, e = paper_split_range("backtesting")
        self.assertEqual(s, date(2024, 1, 1))
        self.assertEqual(e, date(2025, 12, 31))

    def test_yahoo_exclusive_end(self) -> None:
        self.assertEqual(yahoo_period2_exclusive_end(date(2025, 12, 31)), date(2026, 1, 1))

    def test_filter_panel(self) -> None:
        rows = []
        for d in pd.date_range("2023-06-01", "2024-06-30", freq="B"):
            rows.append({"date": d, "ticker": "AAA", "close": 1.0, "open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0})
        df = pd.DataFrame(rows)
        out = filter_panel_to_date_range(df, date(2024, 1, 1), date(2024, 3, 31))
        self.assertTrue(out["date"].min() >= pd.Timestamp("2024-01-01"))
        self.assertTrue(out["date"].max() <= pd.Timestamp("2024-03-31"))

    def test_unique_dates_count(self) -> None:
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])})
        self.assertEqual(count_unique_trading_dates(df), 2)

    def test_expected_counts_table1(self) -> None:
        self.assertEqual(EXPECTED_TRADING_DAYS_SP500["training"], 1763)
        self.assertEqual(EXPECTED_TRADING_DAYS_SP500["live_trading"], 61)

    def test_eval_splits(self) -> None:
        self.assertIn("validation", EVAL_SPLITS)
        self.assertNotIn("training", EVAL_SPLITS)

    def test_split_metadata(self) -> None:
        m = split_metadata("validation")
        self.assertEqual(m["split"], "validation")
        self.assertEqual(m["start_inclusive"], "2023-01-01")
        self.assertEqual(m["end_inclusive"], "2023-12-31")
        self.assertEqual(m["expected_trading_days_sp500_table1"], 250)


if __name__ == "__main__":
    unittest.main()
