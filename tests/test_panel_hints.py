"""Panel calendar hints for orchestration summaries."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from alphacrafter.orchestration.panel_hints import training_panel_diagnostics


class TestPanelHints(unittest.TestCase):
    def test_warns_on_short_panel(self) -> None:
        d = pd.date_range("2023-01-01", periods=100, freq="D")
        rows = []
        for x in d:
            rows.append({"date": x, "ticker": "A", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
        panel = pd.DataFrame(rows)
        out = training_panel_diagnostics(panel)
        self.assertFalse(out.get("empty"))
        self.assertIn("unique_dates_lt_252_ic_noisy", out.get("warnings", []))

    def test_full_span_fewer_warnings(self) -> None:
        d = pd.date_range("2016-01-04", periods=900, freq="D")
        rows = []
        for x in d:
            rows.append({"date": x, "ticker": "A", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
        panel = pd.DataFrame(rows)
        out = training_panel_diagnostics(panel)
        self.assertGreater(int(out["unique_dates"]), 400)
        self.assertNotIn("unique_dates_lt_400_consider_longer_history", out.get("warnings", []))


if __name__ == "__main__":
    unittest.main()
