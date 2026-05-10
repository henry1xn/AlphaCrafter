"""Load Markdown prompt packs shipped with the package."""

from __future__ import annotations

import os
from pathlib import Path


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parent


def load_prompt(name: str, *, max_chars: int | None = None) -> str:
    """
    Load ``{name}.md`` from ``alphacrafter/prompts/``.

    ``name`` without extension, e.g. ``miner_agent``.
    If ``max_chars`` is set, the file is truncated (UTF-8 safe) with a suffix marker.
    """
    p = _prompts_dir() / f"{name}.md"
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8").strip()
    lim = max_chars
    if lim is None:
        env = os.getenv("ALPHACRAFTER_PROMPT_MAX_CHARS", "").strip()
        if env.isdigit():
            lim = int(env)
    if lim and len(text) > lim:
        return text[:lim].rstrip() + "\n\n[...truncated by ALPHACRAFTER_PROMPT_MAX_CHARS...]\n"
    return text
