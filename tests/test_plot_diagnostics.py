"""Smoke test for scripts/plot_diagnostics.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from alphacrafter.memory.shared_memory import SharedMemory, init_database


class TestPlotDiagnosticsScript(unittest.TestCase):
    def test_runs_on_minimal_db(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "plot_diagnostics.py"
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "h.db"
            init_database(db)
            sm = SharedMemory(db)
            try:
                sm.record_factor_event(
                    "f=panel['close']\nfactor=f.pct_change(5).fillna(0)", 0.02, 0.1, "ineffective"
                )
                rid = sm.record_factor_event(
                    "f=panel['close']\nfactor=f.pct_change(3).fillna(0)", 0.04, 0.2, "effective", in_library=True
                )
                h = sm.hash_code(
                    "f=panel['close']\nfactor=f.pct_change(3).fillna(0)",
                )
                eid = sm.insert_ensemble(
                    None,
                    [{"factor_record_id": rid, "code_hash": h, "weight": 1.0, "direction": 1}],
                )
                sm.insert_strategy_candidate(
                    eid,
                    '{"clip":2.5,"gross":1.0,"net_bias":0.0}',
                    0.5,
                    '{"sharpe_ann":0.5,"ann_return_pct":1.0}',
                    "improved",
                )
            finally:
                sm.close()

            out = Path(d) / "plots"
            r = subprocess.run(
                [sys.executable, str(script), "--db", str(db), "--out-dir", str(out)],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            data = json.loads(r.stdout)
            self.assertTrue(data.get("ok"))
            self.assertGreaterEqual(len(data.get("plots", [])), 3)
            self.assertTrue((out / "miner_ic_by_attempt.png").is_file())
            self.assertTrue((out / "trader_sharpe_by_trial.png").is_file())


if __name__ == "__main__":
    unittest.main()
