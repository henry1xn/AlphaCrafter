"""Unit tests for train/OOS factor validation reporting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alphacrafter.agents.default_factors import BUILTIN_FACTOR_CODES
from alphacrafter.agents.miner import MinerAgent
from alphacrafter.data.panel import add_forward_return
from alphacrafter.memory.shared_memory import SharedMemory, init_database
from alphacrafter.reporting.crypto_validation import (
    benchmark_equal_weight_metrics,
    evaluate_library_train_oos,
    factor_long_short_metrics,
)


def _synthetic_long_panel(*, n_dates: int = 45, n_tickers: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = pd.date_range("2018-06-01", periods=n_dates, freq="D")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows: list[dict[str, object]] = []
    for di, d in enumerate(rng):
        for ti, t in enumerate(tickers):
            base = 100.0 + 0.02 * di + 0.5 * ti + (seed % 7) * 0.01
            noise = ((di * 17 + ti * 31 + seed) % 13) * 0.01
            close = float(base + noise)
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": float(1000 + ti * 10),
                }
            )
    panel = pd.DataFrame(rows)
    return add_forward_return(panel)


class TestCryptoValidation(unittest.TestCase):
    def test_evaluate_library_ic_and_optional_sharpe(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "h.db"
            init_database(db)
            sm = SharedMemory(db)
            try:
                sm.ensure_schema()
                code = BUILTIN_FACTOR_CODES[0]
                sm.record_factor_event(code, 0.05, 0.2, "effective", in_library=True)
                rows = sm.list_library_factors()
                self.assertEqual(len(rows), 1)

                train = _synthetic_long_panel(seed=1)
                oos = _synthetic_long_panel(seed=2)
                miner = MinerAgent(sm, asset_class="crypto")
                out = evaluate_library_train_oos(
                    library_rows=rows,
                    miner=miner,
                    train_panel=train,
                    oos_panel=oos,
                    include_sharpe=True,
                )
                self.assertEqual(out["summary"]["n_library_factors"], 1)
                fac = out["factors"][0]
                self.assertIn("ic_train", fac)
                self.assertIn("ic_oos", fac)
                self.assertIsNone(fac.get("err_train"))
                self.assertIsNone(fac.get("err_oos"))
                self.assertIsNotNone(fac.get("ls_sharpe_train"))
            finally:
                sm.close()

    def test_benchmark_not_empty_on_synthetic(self) -> None:
        p = _synthetic_long_panel()
        m = benchmark_equal_weight_metrics(p)
        self.assertGreaterEqual(float(m.get("n", 0.0)), 1.0)

    def test_factor_ls_metrics_runs(self) -> None:
        p = _synthetic_long_panel()
        code = BUILTIN_FACTOR_CODES[0]
        m = factor_long_short_metrics(p, code)
        self.assertIn("sharpe_ann", m)


if __name__ == "__main__":
    unittest.main()
