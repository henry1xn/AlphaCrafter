#!/usr/bin/env python3
"""
因子层验证：在「训练窗」与「样本外窗」上复算因子库 **Z** 的 **IC / IR**（默认不做截面多空 Sharpe、不写等权基准）。

典型用法（与 ``run_alphacrafter.py --split training --miner-only`` 共用同一 ``ALPHACRAFTER_DB_PATH``）::

    conda activate alphacrafter
    cd ~/projects/AlphaCrafter
    python scripts/validate_experiment.py \\
      --crypto-data-dir /path/to/klines \\
      --tickers 20 \\
      --train-split training \\
      --oos-split validation \\
      --json-out runs/factor_ic_report.json \\
      --csv-out runs/factor_ic_ir_table.csv \\
      --markdown-out runs/factor_ic_report.md \\
      --library-json-out runs/factor_library_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alphacrafter.agents.miner import MinerAgent
from alphacrafter.config.settings import DB_PATH, ORCH_TICKER_LIMIT, PANEL_TRADING_DAYS
from alphacrafter.data.universe import load_crypto_universe
from alphacrafter.memory.shared_memory import SharedMemory
from alphacrafter.reporting.crypto_validation import (
    factor_library_rows_to_jsonable,
    ic_sharpe_table_dataframe,
    run_crypto_validation_report,
    write_factor_validation_markdown,
)


def _default_db() -> Path:
    raw = os.getenv("ALPHACRAFTER_DB_PATH", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path(DB_PATH).resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Factor-only validation: train vs OOS IC/IR on library Z; optional LS Sharpe / benchmark."
    )
    p.add_argument("--crypto-data-dir", type=Path, required=True, metavar="DIR", help="K-line root (same as run_alphacrafter)")
    p.add_argument("--tickers", type=int, default=None, help="Universe size (default: ORCH_TICKER_LIMIT)")
    p.add_argument("--crypto-rank-by", choices=["volume", "none"], default=None, help="Crypto ranking (default: env or volume)")
    p.add_argument("--db", type=Path, default=None, help="SQLite shared memory (default ALPHACRAFTER_DB_PATH or data/shared_memory.db)")
    p.add_argument(
        "--train-split",
        type=str,
        default="training",
        help="IC in-sample window: training | validation | backtesting | live_trading (default: training)",
    )
    p.add_argument(
        "--oos-split",
        type=str,
        default="validation",
        help="IC out-of-sample window (default: validation). Common: validation | backtesting",
    )
    p.add_argument("--trading-days", type=int, default=None, help=f"Panel loader span (default {PANEL_TRADING_DAYS})")
    p.add_argument(
        "--with-ls-sharpe",
        action="store_true",
        help="Also compute per-factor long/short annualized Sharpe on train/OOS (strategy-style; off by default)",
    )
    p.add_argument(
        "--with-benchmark",
        action="store_true",
        help="Include equal-weight long-only benchmark metrics on train/OOS panels",
    )
    p.add_argument("--csv-out", type=Path, default=None, help="Write factor-level CSV (IC/IR columns)")
    p.add_argument("--json-out", type=Path, default=None, help="Write full JSON report")
    p.add_argument("--markdown-out", type=Path, default=None, help="Write Markdown summary (tables for IC/IR)")
    p.add_argument(
        "--library-json-out",
        type=Path,
        default=None,
        help="Write snapshot of Z (code + stored ic/ir + hashes) from DB before numerical re-evaluation",
    )
    args = p.parse_args(argv)

    crypto_dir = Path(args.crypto_data_dir).expanduser().resolve()
    if not crypto_dir.is_dir():
        print(json.dumps({"ok": False, "error": "not_a_directory", "path": str(crypto_dir)}), file=sys.stderr)
        return 1

    db_path = Path(args.db).expanduser().resolve() if args.db else _default_db()
    if not db_path.is_file():
        print(json.dumps({"ok": False, "error": "db_not_found", "path": str(db_path)}), file=sys.stderr)
        return 1

    lim = int(args.tickers) if args.tickers is not None else ORCH_TICKER_LIMIT
    rank = (
        str(args.crypto_rank_by).strip().lower()
        if args.crypto_rank_by is not None
        else os.getenv("ALPHACRAFTER_CRYPTO_RANK_BY", "volume").strip().lower()
    )
    td = int(args.trading_days if args.trading_days is not None else PANEL_TRADING_DAYS)

    prev_ac = os.environ.get("ALPHACRAFTER_ASSET_CLASS")
    try:
        os.environ["ALPHACRAFTER_ASSET_CLASS"] = "crypto"
        uni = load_crypto_universe(crypto_dir, ticker_limit=lim, rank_by=rank)
        tickers_list = uni["ticker"].astype(str).tolist()
    finally:
        if prev_ac is None:
            os.environ.pop("ALPHACRAFTER_ASSET_CLASS", None)
        else:
            os.environ["ALPHACRAFTER_ASSET_CLASS"] = prev_ac

    sm = SharedMemory(db_path)
    try:
        rows = sm.list_library_factors()
        if args.library_json_out:
            lib_path = Path(args.library_json_out).expanduser().resolve()
            lib_path.parent.mkdir(parents=True, exist_ok=True)
            snap = {"db_path": str(db_path), "n_factors": len(rows), "factors": factor_library_rows_to_jsonable(rows)}
            lib_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

        miner = MinerAgent(sm, asset_class="crypto")
        report = run_crypto_validation_report(
            crypto_dir=crypto_dir,
            tickers=tickers_list,
            library_rows=rows,
            miner=miner,
            train_split=str(args.train_split),
            oos_split=str(args.oos_split),
            trading_days=td,
            include_sharpe=bool(args.with_ls_sharpe),
            include_benchmark=bool(args.with_benchmark),
        )
        report["db_path"] = str(db_path)
        report["n_tickers_requested"] = lim
        report["report_mode"] = "factor_ic_ir"
        if args.library_json_out:
            report["wrote_library_json"] = str(Path(args.library_json_out).expanduser().resolve())
    finally:
        sm.close()

    if args.json_out:
        outp = Path(args.json_out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["wrote_json"] = str(outp)

    if args.csv_out:
        df = ic_sharpe_table_dataframe(report)
        outc = Path(args.csv_out).expanduser().resolve()
        outc.parent.mkdir(parents=True, exist_ok=True)
        if df.empty:
            outc.write_text("", encoding="utf-8")
        else:
            df.to_csv(outc, index=False)
        report["wrote_csv"] = str(outc)

    if args.markdown_out:
        md_path = Path(args.markdown_out).expanduser().resolve()
        write_factor_validation_markdown(report, md_path)
        report["wrote_markdown"] = str(md_path)

    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
