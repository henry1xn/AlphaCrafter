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
    text = _openai_assistant_text(msg)
    if not text:
        raise RuntimeError(f"LLM empty content: {str(data)[:800]}")
    return text


def _openai_assistant_text(msg: dict[str, Any]) -> str:
    """
    Normalize ``choices[0].message`` to a single string.

    Some OpenAI-compatible providers (e.g. DeepSeek reasoning models) return an empty
    ``content`` but put chain-of-thought + code in ``reasoning_content``; we fall back
    so Miner can still ``extract_python_block``.
    """
    content = msg.get("content")
    if isinstance(content, list):  # multimodal
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        content = "".join(parts)
    if isinstance(content, str) and content.strip():
        return content.strip()
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        alt = msg.get(key)
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    return ""


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


def _maybe_log_llm(
    *,
    agent: str | None,
    provider: str,
    model: str | None,
    system: str,
    user: str,
    response: str,
    max_tokens: int,
) -> None:
    try:
        from alphacrafter.utils.llm_interaction_log import log_llm_turn

        log_llm_turn(
            agent=agent or "unknown",
            provider=provider,
            model=model,
            system=system,
            user=user,
            response=response,
            max_tokens=max_tokens,
        )
    except Exception:  # noqa: BLE001 — logging must never break inference
        pass


def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 1200,
    agent: str | None = None,
) -> str:
    """
    Return assistant plain text.

    ``agent``: logical multi-agent caller for JSONL logs — e.g. ``miner``, ``screener``.

    Backends (via env):

    - **anthropic** — ``ANTHROPIC_API_KEY``, ``ANTHROPIC_MODEL``
    - **openai-compatible** — ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL``
    - **minimax** — ``MINIMAX_API_KEY``, ``MINIMAX_MODEL``, optional ``MINIMAX_API_URL``

    Logging: ``ALPHACRAFTER_LLM_LOG`` (default on), path ``ALPHACRAFTER_LLM_LOG_PATH`` or
    ``<DATA_DIR>/logs/llm_interactions.jsonl``.
    """
    prov = resolve_llm_provider()
    out: str
    model_used: str | None = None

    if prov == "anthropic":
        key = anthropic_api_key()
        if not key or anthropic is None:
            out = _offline_stub_response(user)
            model_used = "stub-offline"
        else:
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
            out = "".join(parts).strip()
            model_used = anthropic_model()

    elif prov == "openai":
        key = _openai_api_key()
        if not key:
            out = _offline_stub_response(user)
            model_used = "stub-offline"
        else:
            model_used = _openai_model()
            out = _chat_completions_openai_format(
                url=_openai_compatible_url(),
                api_key=key,
                model=model_used,
                system=system,
                user=user,
                max_tokens=max_tokens,
            )

    elif prov == "minimax":
        key = _minimax_api_key()
        if not key:
            out = _offline_stub_response(user)
            model_used = "stub-offline"
        else:
            model_used = _minimax_model()
            gid = os.getenv("MINIMAX_GROUP_ID", "").strip()
            extra = {"Group-Id": gid} if gid else None
            out = _chat_completions_openai_format(
                url=_minimax_url(),
                api_key=key,
                model=model_used,
                system=system,
                user=user,
                max_tokens=max_tokens,
                extra_headers=extra,
            )

    else:
        out = _offline_stub_response(user)
        model_used = "stub"

    _maybe_log_llm(
        agent=agent,
        provider=prov,
        model=model_used,
        system=system,
        user=user,
        response=out,
        max_tokens=max_tokens,
    )
    return out


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
