"""OpenAI-compatible message text extraction (reasoning models)."""

from __future__ import annotations

import unittest

from alphacrafter.utils.llm import _openai_assistant_text


class TestOpenaiAssistantText(unittest.TestCase):
    def test_prefers_content(self) -> None:
        t = _openai_assistant_text({"role": "assistant", "content": "  hello  "})
        self.assertEqual(t, "hello")

    def test_falls_back_to_reasoning_content(self) -> None:
        msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "```python\nx=1\n```\n",
        }
        t = _openai_assistant_text(msg)
        self.assertIn("```python", t)

    def test_multimodal_list(self) -> None:
        msg = {"content": [{"type": "text", "text": "abc"}]}
        self.assertEqual(_openai_assistant_text(msg), "abc")


if __name__ == "__main__":
    unittest.main()
