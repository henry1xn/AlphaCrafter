"""Offline tests (no Yahoo) using injected panel + temp SQLite."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from alphacrafter.agents.factor_exec import execute_factor_code
from alphacrafter.data.panel import add_forward_return
from alphacrafter.memory.shared_memory import SharedMemory, init_database
from alphacrafter.metrics.ic import cross_sectional_ic_ir
from alphacrafter.agents.screener import _parse_regime_llm
from alphacrafter.orchestration.loop import run_pipeline


def _synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-06-01", periods=55)
    tickers = [f"T{i}" for i in range(12)]
    rows: list[dict[str, object]] = []
    for dt in dates:
        for t in tickers:
            c = float(50.0 + rng.normal(0, 1.0))
            rows.append(
                {
                    "date": dt,
                    "ticker": t,
                    "open": c,
                    "high": c,
                    "low": c,
                    "close": c,
                    "volume": 1_000_000.0,
                }
            )
    return add_forward_return(pd.DataFrame(rows))


class TestMetricsAndExec(unittest.TestCase):
    def test_ic_ir_finite(self) -> None:
        df = _synthetic_panel()
        df = df.assign(factor=df.groupby("ticker")["close"].pct_change(3))
        df2 = df.dropna(subset=["factor", "fwd_ret"])
        ic, ir = cross_sectional_ic_ir(df2)
        self.assertTrue(ic == ic)
        self.assertTrue(ir == ir)

    def test_execute_factor_code(self) -> None:
        df = _synthetic_panel()
        base = df.drop(columns=["fwd_ret"], errors="ignore")
        code = "g = panel.groupby('ticker')['close']\nfactor = g.pct_change(2).fillna(0.0)\n"
        fac = execute_factor_code(code, base)
        self.assertEqual(len(fac), len(base))


class TestRegimeParse(unittest.TestCase):
    def test_strips_markdown_fence(self) -> None:
        m = {"mkt_vol20": 0.02, "mkt_trend20": 0.0}
        text = "```python\nx = 1\n```\nuptrend\nStrong risk-on breadth."
        label, raw = _parse_regime_llm(text, m)
        self.assertEqual(label, "uptrend")
        self.assertIn("breadth", raw)

    def test_heuristic_when_only_fence(self) -> None:
        m = {"mkt_vol20": 0.02, "mkt_trend20": 0.0}
        label, raw = _parse_regime_llm("```python\npass\n```", m)
        self.assertEqual(label, "high_volatility")


class TestOrchestrationInjectedPanel(unittest.TestCase):
    def test_screener_trader_without_miner(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "h.db"
            init_database(p)
            sm = SharedMemory(p)
            code = "g = panel.groupby('ticker')['close']\nfactor = g.pct_change(2).fillna(0.0)\n"
            sm.record_factor_event(code, 0.05, 0.5, "effective", in_library=True)
            panel = _synthetic_panel()
            try:
                out = run_pipeline(memory=sm, panel=panel, run_miner=False)
            finally:
                sm.close()
            self.assertTrue(out.get("ok"))
            self.assertIsNone(out.get("miner"))
            self.assertIsNotNone(out.get("ensemble_id"))
            self.assertIsNotNone(out.get("trader"))


if __name__ == "__main__":
    unittest.main()
