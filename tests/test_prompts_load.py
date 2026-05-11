"""Prompt packs ship with the package."""

from __future__ import annotations

import unittest

from alphacrafter.prompts.loader import load_prompt


class TestPromptLoader(unittest.TestCase):
    def test_core_prompts_exist(self) -> None:
        for name in (
            "miner_agent",
            "miner_agent_crypto",
            "screener_agent",
            "trader_agent",
            "universal_us_market",
        ):
            text = load_prompt(name)
            self.assertGreater(len(text), 40, msg=name)


if __name__ == "__main__":
    unittest.main()
