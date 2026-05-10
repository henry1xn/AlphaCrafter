"""Trader agent — Algorithm 3: search π, backtest, simulated execution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphacrafter.agents.factor_exec import execute_factor_code
from alphacrafter.agents.screener import EnsembleState
from alphacrafter.backtest.vectorized import backtest_long_short, pivot_close_returns, pivot_signal_from_long
from alphacrafter.config.settings import TRADER_MAX_EXPLORATIONS
from alphacrafter.memory.shared_memory import SharedMemory


@dataclass
class TraderResult:
    best_spec: dict[str, Any]
    best_score: float
    best_metrics: dict[str, float]
    strategy_candidate_id: int | None
    live_result: dict[str, Any]


class TraderAgent:
    def __init__(self, memory: SharedMemory, *, max_explorations: int | None = None) -> None:
        self.memory = memory
        self.max_explorations = int(max_explorations if max_explorations is not None else TRADER_MAX_EXPLORATIONS)

    def _ensemble_signal(
        self,
        panel: pd.DataFrame,
        ensemble: EnsembleState,
    ) -> pd.DataFrame:
        base = panel.drop(columns=["fwd_ret"], errors="ignore").copy()
        acc: pd.DataFrame | None = None
        for m in ensemble.members:
            row = self.memory.get_factor_row(int(m["factor_record_id"]))
            if row is None:
                continue
            code = str(row["code"])
            fac = execute_factor_code(code, base)
            wide = pivot_signal_from_long(base, fac)
            sd = wide.std(axis=1).replace(0, np.nan)
            z = wide.sub(wide.mean(axis=1), axis=0).div(sd, axis=0)
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            piece = z * float(m["weight"]) * int(m["direction"])
            acc = piece if acc is None else acc.add(piece, fill_value=0.0)
        if acc is None:
            return pd.DataFrame()
        return acc.fillna(0.0)

    def construct_strategy(
        self,
        base_signal: pd.DataFrame,
        *,
        clip: float,
        gross: float,
        net_bias: float,
    ) -> dict[str, Any]:
        """Return JSON-serializable strategy spec."""
        return {
            "clip": float(clip),
            "gross": float(gross),
            "net_bias": float(net_bias),
        }

    def apply_strategy_spec(self, signal: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
        clip = float(spec.get("clip", 3.0))
        gross = float(spec.get("gross", 1.0))
        net = float(spec.get("net_bias", 0.0))
        z = signal.clip(lower=-clip, upper=clip) * gross
        if abs(net) > 1e-12:
            z = z.add(net / max(z.shape[1], 1), axis=1)
        return z

    def backtest(self, signal: pd.DataFrame, returns: pd.DataFrame) -> tuple[float, dict[str, float]]:
        _, metrics = backtest_long_short(signal, returns, signal_lag=1)
        return float(metrics.get("sharpe_ann", 0.0)), metrics

    def exploration_terminated(self, trials: int, no_improve: int) -> bool:
        if trials >= self.max_explorations:
            return True
        stall = int(os.getenv("ALPHACRAFTER_TRADER_STALL", "4"))
        return no_improve >= stall

    def live_trading(self, spec: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
        """Placeholder live stage — no broker connectivity."""
        return {
            "mode": "simulated",
            "spec": spec,
            "assumed_daily_mean_from_backtest": metrics.get("mean_daily", 0.0),
        }

    def run(self, panel: pd.DataFrame, ensemble: EnsembleState | None) -> TraderResult | None:
        if ensemble is None or not ensemble.members:
            return None

        returns = pivot_close_returns(panel)
        raw_sig = self._ensemble_signal(panel, ensemble)
        if raw_sig.empty or returns.empty:
            return None

        r_best = float("-inf")
        pi_best: dict[str, Any] | None = None
        metrics_best: dict[str, float] = {}
        cand_id_best: int | None = None

        trials = 0
        no_improve = 0
        clips = [1.5, 2.5, 3.5]
        grosses = [0.65, 0.85, 1.0]
        nets = [-0.05, 0.0, 0.05]

        while True:
            if self.exploration_terminated(trials, no_improve):
                break
            clip = clips[trials % len(clips)]
            gross = grosses[(trials // len(clips)) % len(grosses)]
            net = nets[(trials // (len(clips) * len(grosses))) % len(nets)]
            pi = self.construct_strategy(raw_sig, clip=clip, gross=gross, net_bias=net)
            sig = self.apply_strategy_spec(raw_sig, pi)
            score, metrics = self.backtest(sig, returns)
            trials += 1

            meta = "rejected"
            if score > r_best:
                r_best = score
                pi_best = pi
                metrics_best = metrics
                no_improve = 0
                meta = "improved"
            else:
                no_improve += 1

            spec_json = json.dumps(pi, ensure_ascii=False)
            mid = self.memory.insert_strategy_candidate(
                ensemble.ensemble_id,
                spec_json,
                score,
                json.dumps(metrics, ensure_ascii=False),
                meta,
            )
            if meta == "improved":
                cand_id_best = mid

        if pi_best is None:
            return None

        live = self.live_trading(pi_best, metrics_best)
        self.memory.insert_strategy_execution(
            cand_id_best,
            json.dumps(live, ensure_ascii=False),
            "executed",
        )
        return TraderResult(
            best_spec=pi_best,
            best_score=float(r_best),
            best_metrics=metrics_best,
            strategy_candidate_id=cand_id_best,
            live_result=live,
        )
