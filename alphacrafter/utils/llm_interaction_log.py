"""Append-only JSONL log for multi-agent LLM calls (Miner / Screener)."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from alphacrafter.config.settings import DATA_DIR

_lock = threading.Lock()
_MAX_PREVIEW = int(os.getenv("ALPHACRAFTER_LLM_LOG_PREVIEW_CHARS", "4000") or "4000")


def _default_log_path() -> Path:
    custom = os.getenv("ALPHACRAFTER_LLM_LOG_PATH", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return (DATA_DIR / "logs" / "llm_interactions.jsonl").resolve()


def _trunc(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n...[truncated, total_chars={len(s)}]"


def log_llm_turn(
    *,
    agent: str,
    provider: str,
    model: str | None,
    system: str,
    user: str,
    response: str,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Append one JSON object per line to the log file (UTF-8).

    Disabled when ``ALPHACRAFTER_LLM_LOG=0`` / ``false`` / ``no``.
    Path: ``ALPHACRAFTER_LLM_LOG_PATH`` or ``<DATA_DIR>/logs/llm_interactions.jsonl``.
    """
    flag = os.getenv("ALPHACRAFTER_LLM_LOG", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return

    path = _default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    rec: dict[str, Any] = {
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ts_unix": time.time(),
        "agent": agent,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "system_preview": _trunc(system, _MAX_PREVIEW),
        "user_preview": _trunc(user, _MAX_PREVIEW),
        "response_preview": _trunc(response, _MAX_PREVIEW),
        "response_chars": len(response),
    }
    if extra:
        rec["extra"] = extra

    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(line)
