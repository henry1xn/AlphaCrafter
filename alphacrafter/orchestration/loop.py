"""End-to-end orchestration: Miner → Screener → Trader."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from alphacrafter.agents.default_factors import BUILTIN_FACTOR_CODES
from alphacrafter.agents.miner import MinerAgent
from alphacrafter.agents.screener import ScreenerAgent
from alphacrafter.agents.trader import TraderAgent
from alphacrafter.config.settings import ORCH_TICKER_LIMIT, PANEL_TRADING_DAYS
from alphacrafter.data.panel import add_forward_return, build_long_panel
from alphacrafter.data.universe import load_universe_csv
from alphacrafter.memory.shared_memory import SharedMemory, init_database


def _maybe_seed_default_factors(sm: SharedMemory, miner: MinerAgent, panel: pd.DataFrame) -> dict[str, Any]:
    """
    If Z is still empty (common with short windows / strict IC), try curated factors
    with a slightly relaxed IC floor so Screener/Trader can run end-to-end.
    """
    if os.getenv("ALPHACRAFTER_DISABLE_BUILTIN_SEED", "").strip().lower() in {"1", "true", "yes"}:
        return {"seeded": False, "reason": "disabled"}
    if sm.list_library_factors():
        return {"seeded": False, "reason": "library_nonempty"}
    ratio = float(os.getenv("ALPHACRAFTER_MINER_SEED_IC_RATIO", "0.45"))
    thr = max(0.005, float(miner.ic_accept) * ratio)
    last: dict[str, Any] = {}
    for i, code in enumerate(BUILTIN_FACTOR_CODES):
        ic, ir, err = miner.validate(code, panel)
        last = {"builtin_index": i, "ic": ic, "ir": ir, "err": err}
        if err is None and ic is not None and ic >= thr:
            sm.record_factor_event(code, ic, ir, "effective", in_library=True)
            return {"seeded": True, "builtin_index": i, "ic": ic, "threshold": thr}
    out: dict[str, Any] = {"seeded": False, "tried": len(BUILTIN_FACTOR_CODES), "threshold": thr, "last_try": last}
    return out


def run_pipeline(
    *,
    memory: SharedMemory | None = None,
    universe_csv: str | Path | None = None,
    ticker_limit: int | None = None,
    trading_days: int | None = None,
    panel_sleep: float | None = None,
    panel: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    run_miner: bool = True,
) -> dict[str, Any]:
    """
    One full pass over agents. Downloads OHLCV unless ``panel`` is injected later
    (kept simple: always builds panel here).
    """
    close_owned = memory is None
    if close_owned:
        db_path = init_database()
        sm = SharedMemory(db_path)
    else:
        sm = memory
    try:
        sm.ensure_schema()

        if panel is None or panel.empty:
            uni = load_universe_csv(universe_csv)
            lim = int(ticker_limit if ticker_limit is not None else ORCH_TICKER_LIMIT)
            tickers_list = uni["ticker"].head(lim).astype(str).tolist()
            days = int(trading_days if trading_days is not None else PANEL_TRADING_DAYS)
            sleep = panel_sleep
            if sleep is None:
                sleep = float(os.getenv("ALPHACRAFTER_PANEL_SLEEP", "0"))
            panel = build_long_panel(tickers_list, trading_days=days, sleep_sec=sleep)
        else:
            panel = panel.copy()
            tickers_list = list(tickers) if tickers else sorted(panel["ticker"].astype(str).unique().tolist())

        if panel.empty:
            return {"ok": False, "error": "empty_panel", "tickers": tickers_list}

        miner = MinerAgent(sm)
        miner_summary = miner.run(panel, tickers_list) if run_miner else None
        # miner.run adds fwd_ret on an internal copy only — align pipeline panel for seed / metrics
        panel = add_forward_return(panel)

        seed_meta = _maybe_seed_default_factors(sm, miner, panel)

        screener = ScreenerAgent(sm)
        ensemble, scr_meta = screener.run(panel)

        trader = TraderAgent(sm)
        trade = trader.run(panel, ensemble)

        out: dict[str, Any] = {
            "ok": True,
            "tickers_used": tickers_list,
            "miner": asdict(miner_summary) if miner_summary is not None else None,
            "miner_seed": seed_meta,
            "screener": scr_meta,
            "ensemble_id": getattr(ensemble, "ensemble_id", None),
            "regime": getattr(ensemble, "regime_label", None) if ensemble else None,
        }
        if trade is not None:
            out["trader"] = {
                "best_score": trade.best_score,
                "best_metrics": trade.best_metrics,
                "best_spec": trade.best_spec,
                "live": trade.live_result,
                "strategy_candidate_id": trade.strategy_candidate_id,
            }
        else:
            out["trader"] = None
        return out
    finally:
        if close_owned:
            sm.close()
