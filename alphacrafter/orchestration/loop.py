"""End-to-end orchestration: Miner → Screener → Trader."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from alphacrafter.agents.default_factors import BUILTIN_FACTOR_CODES
from alphacrafter.agents.miner import MinerAgent, MinerRunSummary
from alphacrafter.agents.screener import ScreenerAgent
from alphacrafter.agents.trader import TraderAgent
from alphacrafter.config.settings import ORCH_TICKER_LIMIT, PANEL_TRADING_DAYS
from alphacrafter.data.panel import add_forward_return, build_long_panel, build_long_panel_crypto
from alphacrafter.data.splits import (
    EVAL_SPLITS,
    SplitName,
    count_unique_trading_dates,
    filter_panel_to_date_range,
    normalize_split_name,
    paper_split_range,
    split_metadata,
    yahoo_period2_exclusive_end,
)
from alphacrafter.data.universe import load_crypto_universe, load_universe_csv
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


def _load_panel_for_split(
    tickers_list: list[str],
    phase: SplitName,
    *,
    trading_days: int,
    sleep_sec: float,
    crypto_data_dir: Path | None,
) -> pd.DataFrame:
    start_inc, end_inc = paper_split_range(phase)
    if crypto_data_dir is not None:
        return build_long_panel_crypto(
            tickers_list,
            crypto_data_dir,
            start=start_inc,
            end=end_inc,
            trading_days=trading_days,
            sleep_sec=sleep_sec,
        )
    fetch_end = yahoo_period2_exclusive_end(end_inc)
    raw = build_long_panel(
        tickers_list,
        start=start_inc,
        end=fetch_end,
        trading_days=trading_days,
        sleep_sec=sleep_sec,
    )
    return filter_panel_to_date_range(raw, start_inc, end_inc)


def run_pipeline(
    *,
    memory: SharedMemory | None = None,
    universe_csv: str | Path | None = None,
    crypto_data_dir: str | Path | None = None,
    crypto_rank_by: str | None = None,
    ticker_limit: int | None = None,
    trading_days: int | None = None,
    panel_sleep: float | None = None,
    panel: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    run_miner: bool = True,
    dataset_split: str | None = None,
) -> dict[str, Any]:
    """
    One full pass over agents.

    **Crypto local mode:** pass ``crypto_data_dir`` to load k-lines from CSV/Parquet
    (no Yahoo / no ``sp500_wiki.csv``). Table 1 split discipline unchanged for eval phases.

    **Paper split discipline:** on ``validation`` / ``backtesting`` / ``live_trading`` with
    downloaded panels, Miner runs only on **training** to populate Z; eval window is read-only
    for Z (including builtin seed).
    """
    close_owned = memory is None
    if close_owned:
        db_path = init_database()
        sm = SharedMemory(db_path)
    else:
        sm = memory
    crypto_root = Path(crypto_data_dir).expanduser().resolve() if crypto_data_dir else None
    asset_prev = os.environ.get("ALPHACRAFTER_ASSET_CLASS")
    changed_asset = False
    try:
        if crypto_root is not None:
            os.environ["ALPHACRAFTER_ASSET_CLASS"] = "crypto"
            changed_asset = True

        sm.ensure_schema()
        split_raw = (
            dataset_split
            if dataset_split is not None
            else os.getenv("ALPHACRAFTER_PAPER_SPLIT", "").strip()
        )
        split_name = normalize_split_name(split_raw) if split_raw else None

        days = int(trading_days if trading_days is not None else PANEL_TRADING_DAYS)
        sleep = panel_sleep
        if sleep is None:
            sleep = float(os.getenv("ALPHACRAFTER_PANEL_SLEEP", "0"))

        asset = "crypto" if crypto_root is not None else "equity"
        miner: MinerAgent = MinerAgent(sm, asset_class=asset)
        miner_summary: MinerRunSummary | None = None
        seed_meta: dict[str, Any]
        train_seed_meta: dict[str, Any] | None = None
        library_discipline: dict[str, Any] | None = None

        downloaded = panel is None or panel.empty

        if downloaded:
            lim = int(ticker_limit if ticker_limit is not None else ORCH_TICKER_LIMIT)
            rank = (
                crypto_rank_by
                if crypto_rank_by is not None
                else os.getenv("ALPHACRAFTER_CRYPTO_RANK_BY", "volume").strip().lower()
            )
            if crypto_root is not None:
                uni = load_crypto_universe(crypto_root, ticker_limit=lim, rank_by=rank)
                tickers_list = uni["ticker"].astype(str).tolist()
            else:
                uni = load_universe_csv(universe_csv)
                tickers_list = uni["ticker"].head(lim).astype(str).tolist()

            if split_name and split_name in EVAL_SPLITS:
                train_panel = _load_panel_for_split(
                    tickers_list, "training", trading_days=days, sleep_sec=sleep, crypto_data_dir=crypto_root
                )
                if train_panel.empty:
                    return {"ok": False, "error": "empty_training_panel", "tickers": tickers_list}

                library_discipline = {
                    "mode": "paper_eval",
                    "z_updates_only_on": "training",
                    "eval_phase": split_name,
                    "data_source": "crypto_local" if crypto_root else "yahoo",
                }
                if run_miner:
                    miner_summary = miner.run(train_panel, tickers_list)
                    train_fwd = add_forward_return(train_panel.copy())
                    train_seed_meta = _maybe_seed_default_factors(sm, miner, train_fwd)
                else:
                    train_seed_meta = {"seeded": False, "reason": "run_miner_false"}

                panel = _load_panel_for_split(
                    tickers_list, split_name, trading_days=days, sleep_sec=sleep, crypto_data_dir=crypto_root
                )
                if panel.empty:
                    return {"ok": False, "error": "empty_panel", "tickers": tickers_list}

                panel_observed_td = count_unique_trading_dates(panel)
                panel = add_forward_return(panel)
                seed_meta = {
                    "seeded": False,
                    "reason": "eval_phase_no_Z_writes",
                    "training_phase_seed": train_seed_meta,
                }

            elif split_name == "training":
                panel = _load_panel_for_split(
                    tickers_list, "training", trading_days=days, sleep_sec=sleep, crypto_data_dir=crypto_root
                )
                if panel.empty:
                    return {"ok": False, "error": "empty_panel", "tickers": tickers_list}
                library_discipline = {
                    "mode": "paper_training",
                    "z_updates_on": "training",
                    "data_source": "crypto_local" if crypto_root else "yahoo",
                }
                panel_observed_td = count_unique_trading_dates(panel)
                miner_summary = miner.run(panel, tickers_list) if run_miner else None
                panel = add_forward_return(panel)
                seed_meta = _maybe_seed_default_factors(sm, miner, panel)

            else:
                if crypto_root is not None:
                    panel = build_long_panel_crypto(tickers_list, crypto_root, trading_days=days, sleep_sec=sleep)
                else:
                    panel = build_long_panel(tickers_list, trading_days=days, sleep_sec=sleep)
                if panel.empty:
                    return {"ok": False, "error": "empty_panel", "tickers": tickers_list}
                panel_observed_td = count_unique_trading_dates(panel)
                library_discipline = {
                    "mode": "rolling_window",
                    "data_source": "crypto_local" if crypto_root else "yahoo",
                }
                miner_summary = miner.run(panel, tickers_list) if run_miner else None
                panel = add_forward_return(panel)
                seed_meta = _maybe_seed_default_factors(sm, miner, panel)
        else:
            panel = panel.copy()
            tickers_list = list(tickers) if tickers else sorted(panel["ticker"].astype(str).unique().tolist())
            if panel.empty:
                return {"ok": False, "error": "empty_panel", "tickers": tickers_list}
            panel_observed_td = count_unique_trading_dates(panel)
            library_discipline = {
                "mode": "injected_panel",
                "note": "dataset_split does not auto-fetch training; Miner follows run_miner on injected panel",
            }
            miner_summary = miner.run(panel, tickers_list) if run_miner else None
            panel = add_forward_return(panel)
            seed_meta = _maybe_seed_default_factors(sm, miner, panel)

        screener = ScreenerAgent(sm)
        ensemble, scr_meta = screener.run(panel)

        trader = TraderAgent(sm)
        trade = trader.run(panel, ensemble)

        out: dict[str, Any] = {
            "ok": True,
            "tickers_used": tickers_list,
            "crypto_data_dir": str(crypto_root) if crypto_root else None,
            "dataset_split": split_name,
            "dataset_split_meta": split_metadata(split_name) if split_name else None,
            "panel_observed_trading_days": panel_observed_td,
            "library_discipline": library_discipline,
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
        if changed_asset:
            if asset_prev is None:
                os.environ.pop("ALPHACRAFTER_ASSET_CLASS", None)
            else:
                os.environ["ALPHACRAFTER_ASSET_CLASS"] = asset_prev
        if close_owned:
            sm.close()
