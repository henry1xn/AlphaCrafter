"""Run one orchestrated Miner → Screener → Trader pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install when cwd is repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alphacrafter.orchestration.loop import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AlphaCrafter orchestration (single pass).")
    p.add_argument("--universe", type=Path, default=None, help="Path to universe CSV")
    p.add_argument("--tickers", type=int, default=None, help="Max tickers from universe head")
    p.add_argument("--days", type=int, default=None, help="Approx trading days window")
    p.add_argument("--sleep-panel", type=float, default=None, help="Per-ticker Yahoo delay (seconds)")
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Only print JSON (no human-readable backtest summary lines).",
    )
    args = p.parse_args(argv)

    summary = run_pipeline(
        universe_csv=args.universe,
        ticker_limit=args.tickers,
        trading_days=args.days,
        panel_sleep=args.sleep_panel,
    )
    print(json.dumps(summary, indent=2, default=str))
    if not args.json_only and summary.get("ok"):
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
