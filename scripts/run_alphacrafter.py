"""Run one orchestrated Miner → Screener → Trader pass."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running without editable install when cwd is repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alphacrafter.orchestration.loop import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AlphaCrafter orchestration (single pass).")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Universe CSV (legacy equities). Ignored when --crypto-data-dir is set.",
    )
    src.add_argument(
        "--crypto-data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Folder of per-symbol k-line CSV/Parquet (e.g. BTCUSDT.csv). Implies crypto mode (no Yahoo / no wiki).",
    )
    p.add_argument("--tickers", type=int, default=None, help="Max symbols from universe (after ranking for crypto)")
    p.add_argument(
        "--crypto-rank-by",
        type=str,
        choices=["volume", "none"],
        default=None,
        help="Crypto universe ranking: volume (default) or none (filename order). Market cap not available from OHLCV.",
    )
    p.add_argument("--days", type=int, default=None, help="Approx calendar span for rolling window (no --split)")
    p.add_argument(
        "--sleep-panel",
        type=float,
        default=None,
        help="Per-ticker delay for Yahoo path only (seconds; ignored for local crypto files)",
    )
    p.add_argument(
        "--split",
        type=str,
        default=None,
        metavar="PHASE",
        help=(
            "Paper Table 1 date phase: training | validation | backtesting | live_trading. "
            "When set, uses fixed [start,end] dates; eval phases keep Z frozen to training-only updates."
        ),
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Only print JSON (no human-readable backtest summary lines).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Write summary.json, benchmark_equal_weight_equity.csv, optional trader_best_equity.csv, "
            "and equity_curves.png (needs matplotlib)."
        ),
    )
    args = p.parse_args(argv)

    crypto_dir = args.crypto_data_dir
    if crypto_dir is None:
        envd = os.getenv("ALPHACRAFTER_CRYPTO_DATA_DIR", "").strip()
        if envd:
            crypto_dir = Path(envd)

    summary = run_pipeline(
        universe_csv=args.universe,
        crypto_data_dir=crypto_dir,
        crypto_rank_by=args.crypto_rank_by,
        ticker_limit=args.tickers,
        trading_days=args.days,
        panel_sleep=args.sleep_panel,
        dataset_split=args.split,
        artifacts_dir=args.artifacts_dir,
    )
    print(json.dumps(summary, indent=2, default=str))
    art = summary.get("artifacts")
    if isinstance(art, dict) and art:
        print("\n[Artifacts]", file=sys.stderr)
        for k, v in sorted(art.items()):
            print(f"  {k}: {v}", file=sys.stderr)
    if not args.json_only and summary.get("ok"):
        bench = summary.get("benchmark") or {}
        bm = bench.get("metrics") if isinstance(bench, dict) else None
        if isinstance(bm, dict) and bm.get("n", 0):
            ar = bm.get("ann_return_pct")
            sr = bm.get("sharpe_ann")
            mdd = bm.get("max_drawdown_pct")
            if ar is not None and sr is not None and mdd is not None:
                print(
                    "\n[Benchmark / 等权多头基准] close→close 日收益等权\n"
                    f"  AR(%):   {ar:.4f}\n"
                    f"  SR:      {sr:.4f}\n"
                    f"  MDD(%):  {mdd:.4f}\n"
                )
        tr = summary.get("trader")
        if isinstance(tr, dict) and tr.get("best_metrics"):
            m = tr["best_metrics"]
            ar = m.get("ann_return_pct")
            sr = m.get("sharpe_ann")
            mdd = m.get("max_drawdown_pct")
            if ar is not None and sr is not None and mdd is not None:
                print(
                    "\n[Backtest / 回测] in-sample best (Trader search)\n"
                    f"  AR(%):   {ar:.4f}\n"
                    f"  SR:      {sr:.4f}\n"
                    f"  MDD(%):  {mdd:.4f}\n"
                )
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
