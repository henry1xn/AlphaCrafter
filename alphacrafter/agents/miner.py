"""Miner agent — Algorithm 1: expand Z, log H, maintenance."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from alphacrafter.agents.factor_exec import execute_factor_code
from alphacrafter.config.settings import (
    MINER_IC_ACCEPT,
    MINER_IC_RETAIN,
    MINER_MAX_ITERATIONS,
)
from alphacrafter.data.panel import add_forward_return
from alphacrafter.memory.shared_memory import SharedMemory
from alphacrafter.metrics.ic import cross_sectional_ic_ir
from alphacrafter.prompts.loader import load_prompt
from alphacrafter.utils.llm import complete_text, extract_python_block


@dataclass
class MinerRunSummary:
    iterations: int
    accepted: int
    rejected: int
    deprecated: int


class MinerAgent:
    """LLM factor generation with sandbox exec and IC/IR gate."""

    def __init__(
        self,
        memory: SharedMemory,
        *,
        max_iterations: int | None = None,
        ic_accept: float | None = None,
        ic_retain: float | None = None,
        max_library_factors: int | None = None,
        asset_class: str = "equity",
    ) -> None:
        self.memory = memory
        self.asset_class = (asset_class or "equity").strip().lower()
        self.max_iterations = int(max_iterations if max_iterations is not None else MINER_MAX_ITERATIONS)
        self.ic_accept = float(ic_accept if ic_accept is not None else MINER_IC_ACCEPT)
        self.ic_retain = float(ic_retain if ic_retain is not None else MINER_IC_RETAIN)
        self.max_library_factors = int(
            max_library_factors if max_library_factors is not None else os.getenv("ALPHACRAFTER_MINER_MAX_Z", "16")
        )

    def _panel_for_exec(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Hide forward returns from factor code to avoid lookahead."""
        return panel.drop(columns=["fwd_ret"], errors="ignore").copy()

    def validate(self, code: str, panel: pd.DataFrame) -> tuple[float | None, float | None, str | None]:
        """Return (ic, ir, error). ic/ir None on failure."""
        base = self._panel_for_exec(panel)
        try:
            fac = execute_factor_code(code, base)
            work = base.assign(factor=fac.values)
            if "fwd_ret" in panel.columns:
                work["fwd_ret"] = panel["fwd_ret"].values
            work = work.dropna(subset=["factor", "fwd_ret"])
            if work.empty:
                return None, None, "empty after dropna"
            n_tick = int(work["ticker"].nunique()) if "ticker" in work.columns else 0
            if n_tick > 4:
                min_names = max(4, min(8, n_tick - 1))
            else:
                min_names = max(3, n_tick)
            ic, ir = cross_sectional_ic_ir(work, min_names=min_names)
            if ic != ic:  # NaN
                return None, None, "ic_nan"
            return ic, ir, None
        except Exception as exc:  # noqa: BLE001
            return None, None, str(exc)

    def generate_code(self, tickers: list[str]) -> str:
        hist = self.memory.recent_factor_events(16)
        lines = [f"- {r['outcome_meta']} ic={r['ic']} ir={r['ir']}" for r in hist]
        history = "\n".join(lines) if lines else "(empty)"
        sample = ", ".join(tickers[:40])
        if self.asset_class == "crypto":
            role = load_prompt("miner_agent_crypto")
            if not role.strip():
                role = load_prompt("miner_agent")
        else:
            role = load_prompt("miner_agent")
        universal = ""
        if self.asset_class != "crypto" and os.getenv("ALPHACRAFTER_PROMPT_INCLUDE_UNIVERSAL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }:
            ulim = int(os.getenv("ALPHACRAFTER_UNIVERSAL_PROMPT_CHARS", "6000") or "6000")
            u = load_prompt("universal_us_market", max_chars=ulim)
            if u.strip():
                universal = "\n\n# Global market & workflow context\n" + u.strip()
        sandbox = (
            "You MUST output executable research code as a single ```python fenced block.\n"
            "Sandbox rules: variables np (NumPy) and pd (Pandas) already exist; do NOT use import.\n"
            "DataFrame `panel` has columns date,ticker,open,high,low,close,volume (no fwd_ret).\n"
            "Assign float Series `factor` with one value per row, aligned to `panel` row order.\n"
            "Use groupby('ticker') for time-series ops; stay vectorized."
        )
        system = (
            (role.strip() + universal + "\n\n--- Technical sandbox (must follow) ---\n" + sandbox)
            if role.strip()
            else sandbox
        )
        user = (
            f"Universe sample tickers: {sample}\n"
            f"Recent shared-memory factor attempts:\n{history}\n"
            "Write ONE new factor. Output only a ```python fenced block."
        )
        max_toks = int(os.getenv("ALPHACRAFTER_MINER_MAX_TOKENS", "1400") or "1400")
        raw = complete_text(system, user, max_tokens=max_toks)
        return extract_python_block(raw)

    def _termination(self, iterations: int, accepted_this_run: int) -> bool:
        if iterations >= self.max_iterations:
            return True
        if len(self.memory.list_library_factors()) >= self.max_library_factors:
            return True
        if accepted_this_run >= 2 and iterations >= 2:
            # optional early stop if enough accepted
            return bool(int(os.getenv("ALPHACRAFTER_MINER_EARLY_STOP", "0")))
        return False

    def run(self, panel: pd.DataFrame, tickers: list[str]) -> MinerRunSummary:
        if panel.empty:
            raise ValueError("Miner received empty panel")
        # Local working copy; callers should run ``add_forward_return`` on their panel for downstream agents.
        panel = add_forward_return(panel)
        iterations = 0
        accepted = 0
        rejected = 0
        deprecated = 0

        while not self._termination(iterations, accepted):
            iterations += 1
            try:
                code = self.generate_code(tickers)
            except Exception as exc:  # noqa: BLE001
                self.memory.record_factor_event(f"# generate failed\n{exc}", None, None, "ineffective")
                rejected += 1
                continue

            ic, ir, err = self.validate(code, panel)
            if err is not None:
                self.memory.record_factor_event(code, ic, ir, "ineffective")
                rejected += 1
                continue

            assert ic is not None and ir is not None
            if ic > self.ic_accept:
                self.memory.record_factor_event(code, ic, ir, "effective", in_library=True)
                accepted += 1
            else:
                self.memory.record_factor_event(code, ic, ir, "ineffective")
                rejected += 1

        # Maintenance — revalidate current library (snapshot by hash)
        library_rows = self.memory.list_library_factors()
        seen: set[str] = set()
        for row in library_rows:
            h = str(row["code_hash"])
            if h in seen:
                continue
            seen.add(h)
            code = str(row["code"])
            ic2, ir2, err2 = self.validate(code, panel)
            if err2 is not None or ic2 is None or ic2 < self.ic_retain:
                self.memory.deactivate_factor_library_rows(h)
                self.memory.record_factor_event(
                    code,
                    ic2,
                    ir2,
                    "deprecated",
                    in_library=False,
                )
                deprecated += 1

        return MinerRunSummary(
            iterations=iterations,
            accepted=accepted,
            rejected=rejected,
            deprecated=deprecated,
        )
