"""Backtest metrics helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alphacrafter.backtest.vectorized import backtest_long_short, daily_portfolio_metrics
from alphacrafter.reporting.artifacts import write_pipeline_artifacts


class TestDailyPortfolioMetrics(unittest.TestCase):
    def test_matches_backtest_long_short(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2022-01-01", periods=80)
        cols = ["A", "B", "C"]
        sig = pd.DataFrame(rng.normal(0, 1, (len(idx), len(cols))), index=idx, columns=cols)
        ret = pd.DataFrame(rng.normal(0.0002, 0.01, (len(idx), len(cols))), index=idx, columns=cols)
        port, m1 = backtest_long_short(sig, ret, signal_lag=1)
        m2 = daily_portfolio_metrics(port)
        for k in m1:
            self.assertAlmostEqual(float(m1[k]), float(m2[k]), places=10, msg=k)


class TestWriteArtifacts(unittest.TestCase):
    def test_writes_benchmark_csv_and_summary(self) -> None:
        from alphacrafter.agents.trader import TraderResult

        dates = pd.bdate_range("2023-01-01", periods=20)
        rows = []
        for d in dates:
            for t in ("X", "Y"):
                rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 100.0,
                        "volume": 1.0,
                    }
                )
        panel = pd.DataFrame(rows)
        summary = {"ok": True, "benchmark": {"metrics": {"n": 1.0}}, "trader": None}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "art"
            paths = write_pipeline_artifacts(root, summary, panel, None)
            self.assertTrue(Path(paths["benchmark_equal_weight_equity_csv"]).is_file())
            self.assertTrue(Path(paths["summary_json"]).is_file())

        tr = TraderResult(
            best_spec={"clip": 1.0},
            best_score=1.0,
            best_metrics={"sharpe_ann": 0.1},
            strategy_candidate_id=1,
            live_result={},
            equity_curve=[
                {"date": "2023-01-03", "daily_ret": 0.01, "equity": 1.01},
                {"date": "2023-01-04", "daily_ret": -0.005, "equity": 1.005},
            ],
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "art2"
            paths = write_pipeline_artifacts(root, summary, panel, tr)
            self.assertIn("trader_best_equity_csv", paths)


if __name__ == "__main__":
    unittest.main()
