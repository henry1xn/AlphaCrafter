"""LLM interaction JSONL log (stub provider)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path


class TestLlmInteractionLog(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("ALPHACRAFTER_LLM_LOG_PATH", None)

    def test_stub_complete_text_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "llm.jsonl"
            os.environ["ALPHACRAFTER_LLM_PROVIDER"] = "stub"
            os.environ["ALPHACRAFTER_LLM_LOG_PATH"] = str(log_path)
            os.environ["ALPHACRAFTER_LLM_LOG"] = "1"
            from alphacrafter.utils.llm import complete_text

            out = complete_text("system prompt", "user body", max_tokens=64, agent="miner")
            self.assertTrue(out.strip())
            self.assertTrue(log_path.is_file())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 1)
            row = json.loads(lines[-1])
            self.assertEqual(row.get("agent"), "miner")
            self.assertEqual(row.get("provider"), "stub")


if __name__ == "__main__":
    unittest.main()
