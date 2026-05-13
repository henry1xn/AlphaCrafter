#!/usr/bin/env python3
"""
从 SQLite 共享内存 H 生成诊断图：Miner 迭代 IC、IC–IR 散点、Trader 试探夏普轨迹、
因子库在训练窗 vs OOS 窗的 IC 散点；可选组合策略在训练窗 vs OOS 窗的夏普对比。

用法见 README「诊断图」一节。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alphacrafter.agents.trader import TraderAgent
from alphacrafter.agents.screener import EnsembleState
from alphacrafter.backtest.vectorized import backtest_long_short, pivot_close_returns
from alphacrafter.config.settings import DB_PATH, PANEL_TRADING_DAYS
from alphacrafter.data.panel import add_forward_return, build_long_panel_crypto
from alphacrafter.data.splits import SplitName, paper_split_range
from alphacrafter.data.universe import load_crypto_universe
from alphacrafter.memory.shared_memory import SharedMemory
from alphacrafter.agents.miner import MinerAgent


def _default_db() -> Path:
    raw = os.getenv("ALPHACRAFTER_DB_PATH", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path(DB_PATH).resolve()


def _load_factor_tail(conn: sqlite3.Connection, tail: int) -> pd.DataFrame:
    lim = int(max(tail, 1))
    df = pd.read_sql_query(
        """
        SELECT id, ic, ir, outcome_meta, in_library, created_at
        FROM factor_records
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(lim,),
    )
    if df.empty:
        return df
    return df.sort_values("id").reset_index(drop=True)


def _load_latest_ensemble(conn: sqlite3.Connection) -> tuple[int | None, list[dict] | None]:
    cur = conn.execute("SELECT id, members_json FROM ensembles ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if row is None:
        return None, None
    eid = int(row[0])
    members = json.loads(row[1])
    return eid, members


def _load_best_spec_for_ensemble(conn: sqlite3.Connection, ensemble_id: int) -> dict | None:
    cur = conn.execute(
        """
        SELECT spec_json FROM strategy_candidates
        WHERE ensemble_id = ?
        ORDER BY CASE WHEN backtest_score IS NULL THEN 1 ELSE 0 END,
                 backtest_score DESC
        LIMIT 1
        """,
        (ensemble_id,),
    )
    row = cur.fetchone()
    if row is None or not row[0]:
        return None
    return json.loads(row[0])


def _load_strategy_trials(conn: sqlite3.Connection, ensemble_id: int | None) -> pd.DataFrame:
    if ensemble_id is None:
        return pd.DataFrame()
    cur = conn.execute(
        """
        SELECT id, backtest_score, backtest_metrics_json, exploration_meta
        FROM strategy_candidates
        WHERE ensemble_id = ?
        ORDER BY id ASC
        """,
        (ensemble_id,),
    )
    rows = cur.fetchall()
    out: list[dict] = []
    for i, r in enumerate(rows):
        mid, score, mjson, meta = r
        sharpe = float(score) if score is not None and score == score else float("nan")
        ann = float("nan")
        if mjson:
            try:
                m = json.loads(mjson)
                sharpe = float(m.get("sharpe_ann", sharpe))
                ann = float(m.get("ann_return_pct", float("nan")))
            except json.JSONDecodeError:
                pass
        out.append(
            {
                "trial": i + 1,
                "id": int(mid),
                "sharpe_ann": sharpe,
                "ann_return_pct": ann,
                "exploration_meta": str(meta or ""),
            }
        )
    return pd.DataFrame(out)


def plot_miner_ic_series(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    x = np.arange(1, len(df) + 1)
    ic = pd.to_numeric(df["ic"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, ic, "o-", markersize=3, linewidth=1, label="IC per attempt")
    cmax = ic.cummax()
    ax.plot(x, cmax, "--", color="tab:orange", linewidth=1.5, label="cumulative max IC")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Miner attempt index (chronological in tail window)")
    ax.set_ylabel("IC")
    ax.set_title("Miner: IC along attempts (recent tail)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_miner_ic_ir_scatter(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ic = pd.to_numeric(df["ic"], errors="coerce")
    ir = pd.to_numeric(df["ir"], errors="coerce")
    meta = df["outcome_meta"].astype(str)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, color in (
        ("effective", "tab:green"),
        ("ineffective", "tab:gray"),
        ("deprecated", "tab:red"),
    ):
        m = meta == label
        if m.any():
            ax.scatter(ic[m], ir[m], s=22, alpha=0.75, label=label, c=color)
    ax.axhline(0, color="k", linewidth=0.3)
    ax.axvline(0, color="k", linewidth=0.3)
    ax.set_xlabel("IC")
    ax.set_ylabel("IR")
    ax.set_title("Miner: IC vs IR (by outcome)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trader_trials(trials: pd.DataFrame, out_path: Path) -> None:
    if trials.empty or "sharpe_ann" not in trials.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    s = trials["sharpe_ann"].replace([np.inf, -np.inf], np.nan)
    ax.plot(trials["trial"], s, "o-", markersize=4, label="Sharpe (trial)")
    ax.plot(trials["trial"], s.cummax(), "--", color="tab:orange", linewidth=1.5, label="best-so-far")
    ax.set_xlabel("Trader exploration trial (insert order)")
    ax.set_ylabel("Sharpe (annualized)")
    ax.set_title("Trader: Sharpe over optimization trials (eval window backtest)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_factor_ic_train_oos(
    *,
    sm: SharedMemory,
    miner: MinerAgent,
    tickers: list[str],
    crypto_dir: Path,
    oos_split: SplitName,
    trading_days: int,
    out_path: Path,
) -> None:
    rows = sm.list_library_factors()
    if not rows:
        return
    t0, t1 = paper_split_range("training")
    v0, v1 = paper_split_range(oos_split)
    train = build_long_panel_crypto(
        tickers, crypto_dir, start=t0, end=t1, trading_days=trading_days
    )
    oos = build_long_panel_crypto(
        tickers, crypto_dir, start=v0, end=v1, trading_days=trading_days
    )
    train = add_forward_return(train)
    oos = add_forward_return(oos)
    if train.empty or oos.empty:
        return

    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        code = str(r["code"])
        ic_tr, _, etr = miner.validate(code, train)
        ic_oos, _, eo = miner.validate(code, oos)
        if etr is not None or ic_tr != ic_tr or eo is not None or ic_oos != ic_oos:
            continue
        xs.append(float(ic_tr))
        ys.append(float(ic_oos))

    if len(xs) < 1:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.array(xs), np.array(ys), s=36, alpha=0.8, edgecolors="k", linewidths=0.3)
    lim = (-0.15, 0.15)
    ax.plot(lim, lim, "k--", linewidth=0.8, label="y=x")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("IC in-sample (training window)")
    ax.set_ylabel(f"IC out-of-sample ({oos_split})")
    ax.set_title("Factor library: train IC vs OOS IC")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_portfolio_sharpe_train_oos(
    *,
    sm: SharedMemory,
    ensemble_id: int,
    members: list[dict],
    spec: dict,
    tickers: list[str],
    crypto_dir: Path,
    oos_split: SplitName,
    trading_days: int,
    out_path: Path,
) -> None:
    t0, t1 = paper_split_range("training")
    v0, v1 = paper_split_range(oos_split)
    train = add_forward_return(
        build_long_panel_crypto(tickers, crypto_dir, start=t0, end=t1, trading_days=trading_days)
    )
    oos = add_forward_return(
        build_long_panel_crypto(tickers, crypto_dir, start=v0, end=v1, trading_days=trading_days)
    )
    if train.empty or oos.empty:
        return

    trader = TraderAgent(sm)
    ens = EnsembleState(
        ensemble_id=ensemble_id,
        regime_id=0,
        regime_label="",
        members=members,
    )

    def _sharpe(panel: pd.DataFrame) -> float:
        ret = pivot_close_returns(panel)
        raw = trader._ensemble_signal(panel, ens)
        if raw.empty or ret.empty:
            return float("nan")
        sig = trader.apply_strategy_spec(raw, spec)
        _, m = backtest_long_short(sig, ret, signal_lag=1)
        return float(m.get("sharpe_ann", float("nan")))

    sr_tr = _sharpe(train)
    sr_oos = _sharpe(oos)
    if not (np.isfinite(sr_tr) and np.isfinite(sr_oos)):
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([sr_tr], [sr_oos], s=120, zorder=3, edgecolors="k", linewidths=0.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    lo = min(-0.5, sr_tr, sr_oos, -0.1)
    hi = max(0.5, sr_tr, sr_oos, 0.1)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Portfolio Sharpe (training, same ensemble+spec)")
    ax.set_ylabel(f"Portfolio Sharpe ({oos_split}, same ensemble+spec)")
    ax.set_title("Strategy: in-sample vs OOS Sharpe (single point)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot Miner/Trader/factor diagnostics from SQLite H.")
    p.add_argument("--db", type=Path, default=None, help="SQLite path (default: ALPHACRAFTER_DB_PATH or data/shared_memory.db)")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for PNG outputs")
    p.add_argument("--tail", type=int, default=500, help="Last N factor_records rows for Miner plots")
    p.add_argument("--crypto-data-dir", type=Path, default=None, help="If set: IC train vs OOS + optional portfolio Sharpe train/OOS")
    p.add_argument("--tickers", type=int, default=20, help="Universe size for crypto panels when --crypto-data-dir set")
    p.add_argument("--crypto-rank-by", choices=["volume", "none"], default="volume")
    p.add_argument(
        "--oos-split",
        type=str,
        default="backtesting",
        choices=["validation", "backtesting"],
        help="OOS calendar for factor IC scatter and portfolio Sharpe OOS axis",
    )
    p.add_argument("--trading-days", type=int, default=PANEL_TRADING_DAYS, help="Panel load safety span (see build_long_panel_crypto)")
    args = p.parse_args(argv)

    db_path = Path(args.db).expanduser().resolve() if args.db else _default_db()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    written: list[str] = []

    fac = _load_factor_tail(conn, args.tail)
    if not fac.empty:
        p1 = out_dir / "miner_ic_by_attempt.png"
        plot_miner_ic_series(fac, p1)
        written.append(str(p1))
        p2 = out_dir / "miner_ic_vs_ir_scatter.png"
        plot_miner_ic_ir_scatter(fac, p2)
        written.append(str(p2))

    eid, members = _load_latest_ensemble(conn)
    trials = _load_strategy_trials(conn, eid)
    if not trials.empty:
        p3 = out_dir / "trader_sharpe_by_trial.png"
        plot_trader_trials(trials, p3)
        written.append(str(p3))

    conn.close()

    if args.crypto_data_dir is not None:
        crypto_dir = Path(args.crypto_data_dir).expanduser().resolve()
        if not crypto_dir.is_dir():
            print(f"Not a directory: {crypto_dir}", file=sys.stderr)
        else:
            prev_ac = os.environ.get("ALPHACRAFTER_ASSET_CLASS")
            try:
                os.environ["ALPHACRAFTER_ASSET_CLASS"] = "crypto"
                uni = load_crypto_universe(
                    crypto_dir, ticker_limit=int(args.tickers), rank_by=str(args.crypto_rank_by)
                )
                tickers_list = uni["ticker"].astype(str).tolist()
                sm = SharedMemory(db_path)
                try:
                    miner = MinerAgent(sm, asset_class="crypto")
                    oos_split: SplitName = args.oos_split  # type: ignore[assignment]
                    p4 = out_dir / f"factor_ic_train_vs_{args.oos_split}.png"
                    plot_factor_ic_train_oos(
                        sm=sm,
                        miner=miner,
                        tickers=tickers_list,
                        crypto_dir=crypto_dir,
                        oos_split=oos_split,
                        trading_days=int(args.trading_days),
                        out_path=p4,
                    )
                    if p4.is_file():
                        written.append(str(p4))

                    cx = sqlite3.connect(str(db_path))
                    try:
                        spec = _load_best_spec_for_ensemble(cx, eid) if eid is not None else None
                    finally:
                        cx.close()
                    if eid is not None and members and spec:
                        p5 = out_dir / f"portfolio_sharpe_train_vs_{args.oos_split}.png"
                        plot_portfolio_sharpe_train_oos(
                            sm=sm,
                            ensemble_id=eid,
                            members=members,
                            spec=spec,
                            tickers=tickers_list,
                            crypto_dir=crypto_dir,
                            oos_split=oos_split,
                            trading_days=int(args.trading_days),
                            out_path=p5,
                        )
                        if p5.is_file():
                            written.append(str(p5))
                finally:
                    sm.close()
            finally:
                if prev_ac is None:
                    os.environ.pop("ALPHACRAFTER_ASSET_CLASS", None)
                else:
                    os.environ["ALPHACRAFTER_ASSET_CLASS"] = prev_ac

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "plots": written}, indent=2, ensure_ascii=False))
    if not written:
        print("(no plots: empty factor tail and no trader trials; add --crypto-data-dir for factor IC)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
