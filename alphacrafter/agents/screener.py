"""Screener agent — Algorithm 2: regime, suitability, ensemble E_t."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphacrafter.config.settings import SCREENER_MIN_FACTORS, SCREENER_TOP_K
from alphacrafter.memory.shared_memory import SharedMemory
from alphacrafter.prompts.loader import load_prompt
from alphacrafter.utils.llm import complete_text

_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_BAD_FIRST = frozenset(
    {"the", "a", "an", "regime", "market", "label", "line", "this", "here", "python", "json", "plain"}
)


def _parse_regime_llm(text: str, m: dict[str, float]) -> tuple[str, str]:
    """Strip fences / junk; pick first snake_case line; else heuristic from M."""
    cleaned = _FENCE_RE.sub("", text or "").strip()
    lines: list[str] = []
    for ln in cleaned.splitlines():
        s = ln.strip()
        if not s or s.startswith("`"):
            continue
        low = s.lower()
        if low in {"python", "json", "text", "plaintext"}:
            continue
        lines.append(s)

    label = "unknown_regime"
    idx = 0
    for i, s in enumerate(lines):
        if _LABEL_RE.match(s) and "```" not in s:
            label = s.lower()
            idx = i + 1
            break
        if ":" in s:
            tail = s.split(":", 1)[1].strip().split()[0] if s.split(":", 1)[1].strip() else ""
            if tail and _LABEL_RE.match(tail):
                label = tail.lower()
                idx = i + 1
                break
        toks = s.split()
        if toks:
            cand = toks[0]
            if cand.lower() not in _BAD_FIRST and _LABEL_RE.match(cand):
                label = cand.lower()
                idx = i + 1
                break

    raw = "\n".join(lines[idx:]).strip() if lines else ""
    if label == "unknown_regime" or "`" in label:
        v = float(m.get("mkt_vol20") or 0.0)
        t20 = float(m.get("mkt_trend20") or 0.0)
        if v > 0.012:
            label = "high_volatility"
        elif t20 < -0.02:
            label = "downtrend"
        elif t20 > 0.02:
            label = "uptrend"
        else:
            label = "range_bound"
        if not raw:
            raw = "heuristic_fallback"
    return label, raw if raw else cleaned[:800]


@dataclass
class EnsembleState:
    ensemble_id: int
    regime_id: int
    regime_label: str
    members: list[dict[str, Any]]


class ScreenerAgent:
    def __init__(
        self,
        memory: SharedMemory,
        *,
        min_factors: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.memory = memory
        self.min_factors = int(min_factors if min_factors is not None else SCREENER_MIN_FACTORS)
        self.top_k = int(top_k if top_k is not None else SCREENER_TOP_K)

    def market_state(self, panel: pd.DataFrame) -> dict[str, float]:
        """Aggregate M: equal-weight market proxy from panel closes."""
        if panel.empty:
            return {"mkt_vol20": float("nan"), "mkt_trend20": float("nan")}
        mkt = panel.groupby("date", sort=False)["close"].mean()
        ret = mkt.pct_change()
        vol20 = float(ret.iloc[-20:].std(ddof=1)) if len(ret) >= 20 else float(ret.std(ddof=1))
        trend20 = float(mkt.iloc[-1] / mkt.iloc[-20] - 1.0) if len(mkt) >= 20 else 0.0
        return {"mkt_vol20": vol20, "mkt_trend20": trend20}

    def assess_regime(self, m: dict[str, float], memory: SharedMemory) -> tuple[str, str]:
        """LLM regime label + short rationale; heuristic fallback on failure."""
        hist = memory.recent_factor_events(8)
        tail = "\n".join(f"{r['outcome_meta']}" for r in hist) or "(none)"
        pack = load_prompt("screener_agent").strip()
        tail = (
            "You assess the equity market regime. Reply in plain text only — no markdown, no code fences. "
            "Line 1: one snake_case label (letters, digits, underscore). "
            "Line 2: one short rationale sentence."
        )
        system = (pack + "\n\n--- Regime labeling task ---\n" + tail) if pack else tail
        user = (
            f"Market features JSON: {json.dumps(m, ensure_ascii=False)}\n"
            f"Recent memory outcomes:\n{tail}\n"
            "Line 1 label, line 2 rationale."
        )
        try:
            text = complete_text(system, user, max_tokens=200)
            label, raw = _parse_regime_llm(text, m)
        except Exception:  # noqa: BLE001
            label, raw = _parse_regime_llm("", m)
            raw = f"{raw}; llm_failed"
        return label, raw

    def suitability(self, factor_row: Any, m: dict[str, float], regime: str, memory: SharedMemory) -> float:
        """Scalar score s(f, M, R_hat, H) — lightweight blend for Phase 3."""
        ic = float(factor_row["ic"] or 0.0)
        ir = float(factor_row["ir"] or 0.0)
        if not math.isfinite(ic):
            ic = 0.0
        if not math.isfinite(ir):
            ir = 0.0
        base = max(ic, 0.0) * 0.75 + max(min(ir, 3.0), -3.0) / 3.0 * 0.25
        # mild regime tilt: reward contrarian-ish factors in high vol (proxy via code text)
        code = str(factor_row["code"]).lower()
        if "high_vol" in regime or (m.get("mkt_vol20") or 0) > 0.012:
            if "vol" in code or "rank" in code:
                base *= 1.05
        _ = memory
        return float(base)

    def diversify(self, candidates: list[tuple[Any, float]]) -> list[Any]:
        """Greedy top-k with simple bucket cap on code hash modulo."""
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        picked: list[Any] = []
        buckets: dict[int, int] = {}
        for row, s in candidates:
            b = hash(str(row["code_hash"])) % 6
            if buckets.get(b, 0) >= max(2, self.top_k // 3):
                continue
            picked.append(row)
            buckets[b] = buckets.get(b, 0) + 1
            if len(picked) >= self.top_k:
                break
        return picked

    def assign_weight_and_direction(self, row: Any, s: float, regime: str) -> tuple[float, int]:
        w = max(float(s), 1e-8)
        direction = 1
        if "low_vol" in regime and "mom" in str(row["code"]).lower():
            direction = -1
        _ = regime
        return w, direction

    def run(self, panel: pd.DataFrame) -> tuple[EnsembleState | None, dict[str, Any]]:
        rows = self.memory.list_library_factors()
        if len(rows) < self.min_factors:
            return None, {"reason": "insufficient_factors", "have": len(rows), "need": self.min_factors}

        m = self.market_state(panel)
        regime_label, raw = self.assess_regime(m, self.memory)
        regime_id = self.memory.insert_market_regime(regime_label, features=m, raw_assessment=raw)

        scored: list[tuple[Any, float]] = []
        for r in rows:
            s = self.suitability(r, m, regime_label, self.memory)
            scored.append((r, s))

        selected = self.diversify(scored)
        score_by_id = {int(r["id"]): s for r, s in scored}
        sum_w = sum(
            max(self.assign_weight_and_direction(r, score_by_id[int(r["id"])], regime_label)[0], 1e-8)
            for r in selected
        )
        members: list[dict[str, Any]] = []
        for r in selected:
            s = score_by_id[int(r["id"])]
            w0, d0 = self.assign_weight_and_direction(r, s, regime_label)
            w = w0 / sum_w if sum_w > 0 else 0.0
            members.append(
                {
                    "factor_record_id": int(r["id"]),
                    "code_hash": str(r["code_hash"]),
                    "weight": float(w),
                    "direction": int(d0),
                }
            )

        eid = self.memory.insert_ensemble(regime_id, members)
        return (
            EnsembleState(ensemble_id=eid, regime_id=regime_id, regime_label=regime_label, members=members),
            {"market": m},
        )
