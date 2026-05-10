"""Multi-provider LLM completion (Anthropic, OpenAI-compatible, MiniMax) + offline stub."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

_DEFAULT_UA = "AlphaCrafter/1.0 (+https://example.invalid; llm client)"


def anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()


def anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def _openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "deepseek-chat").strip()


def _openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def _minimax_url() -> str:
    return os.getenv(
        "MINIMAX_API_URL",
        "https://api.minimax.io/v1/text/chatcompletion_v2",
    ).strip()


def _minimax_model() -> str:
    return os.getenv("MINIMAX_MODEL", "MiniMax-M2.7").strip()


def _minimax_api_key() -> str:
    return os.getenv("MINIMAX_API_KEY", "").strip()


def resolve_llm_provider() -> str:
    """
    Provider slug: anthropic | openai | minimax | stub.

    If ``ALPHACRAFTER_LLM_PROVIDER`` is unset, pick the first backend with credentials.
    """
    explicit = os.getenv("ALPHACRAFTER_LLM_PROVIDER", "").strip().lower()
    if explicit in {"stub", "offline", "none", "disabled"}:
        return "stub"
    if explicit in {"anthropic", "claude"}:
        return "anthropic"
    if explicit in {"openai", "openai_compatible", "deepseek", "azure_openai"}:
        return "openai"
    if explicit in {"minimax", "mini_max"}:
        return "minimax"

    if anthropic_api_key() and anthropic is not None:
        return "anthropic"
    if _openai_api_key():
        return "openai"
    if _minimax_api_key():
        return "minimax"
    return "stub"


def _chat_completions_openai_format(
    *,
    url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    extra_headers: dict[str, str] | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _DEFAULT_UA,
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(os.getenv("ALPHACRAFTER_LLM_TEMPERATURE", "0.2")),
    }
    timeout = float(os.getenv("ALPHACRAFTER_LLM_TIMEOUT_SEC", "120"))
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {str(data)[:800]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # multimodal
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM empty content: {str(data)[:800]}")
    return content.strip()


def _openai_compatible_url() -> str:
    base = _openai_base_url()
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_python_block(text: str) -> str:
    """Take first ```python ... ``` or ``` ... ``` fence, else whole string stripped."""
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def complete_text(system: str, user: str, *, max_tokens: int = 1200) -> str:
    """
    Return assistant plain text.

    Backends (via env):

    - **anthropic** — ``ANTHROPIC_API_KEY``, ``ANTHROPIC_MODEL``
    - **openai-compatible** — ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` (DeepSeek: https://api.deepseek.com), ``OPENAI_MODEL``
    - **minimax** — ``MINIMAX_API_KEY``, ``MINIMAX_MODEL``, optional ``MINIMAX_API_URL``

    Override auto-selection with ``ALPHACRAFTER_LLM_PROVIDER=anthropic|openai|minimax|stub``.
    """
    prov = resolve_llm_provider()
    if prov == "anthropic":
        key = anthropic_api_key()
        if not key or anthropic is None:
            return _offline_stub_response(user)
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=anthropic_model(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for b in msg.content:
            if getattr(b, "type", None) == "text":
                parts.append(b.text)
        return "".join(parts).strip()

    if prov == "openai":
        key = _openai_api_key()
        if not key:
            return _offline_stub_response(user)
        return _chat_completions_openai_format(
            url=_openai_compatible_url(),
            api_key=key,
            model=_openai_model(),
            system=system,
            user=user,
            max_tokens=max_tokens,
        )

    if prov == "minimax":
        key = _minimax_api_key()
        if not key:
            return _offline_stub_response(user)
        gid = os.getenv("MINIMAX_GROUP_ID", "").strip()
        extra = {"Group-Id": gid} if gid else None
        return _chat_completions_openai_format(
            url=_minimax_url(),
            api_key=key,
            model=_minimax_model(),
            system=system,
            user=user,
            max_tokens=max_tokens,
            extra_headers=extra,
        )

    return _offline_stub_response(user)


def _offline_stub_response(user: str) -> str:
    """Tiny rotating code templates for CI / demos without API keys."""
    seed = sum(ord(c) for c in user) % 3
    templates = [
        (
            "```python\n"
            "# no import — np/pd are injected\n"
            "g = panel.groupby('ticker')['close']\n"
            "mom = g.pct_change(5)\n"
            "factor = -mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
            "```"
        ),
        (
            "```python\n"
            "vol = panel.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(10).std())\n"
            "factor = (-vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
            "```"
        ),
        (
            "```python\n"
            "rank_vol = panel.groupby('date')['volume'].rank(pct=True)\n"
            "factor = (0.5 - rank_vol).fillna(0.0)\n"
            "```"
        ),
    ]
    return templates[seed]
