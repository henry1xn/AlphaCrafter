"""Write CSV/JSON/PNG after ``run_pipeline`` (benchmark + optional trader curve)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from alphacrafter.agents.trader import TraderResult
from alphacrafter.backtest.vectorized import pivot_close_returns


def write_pipeline_artifacts(
    art_dir: str | Path,
    summary: dict[str, Any],
    panel: pd.DataFrame,
    trade: TraderResult | None,
) -> dict[str, str]:
    root = Path(art_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    ret_w = pivot_close_returns(panel)
    bench_r = ret_w.mean(axis=1).dropna() if not ret_w.empty else pd.Series(dtype=float)
    if not bench_r.empty:
        bench_df = pd.DataFrame(
            {
                "date": [pd.Timestamp(t).strftime("%Y-%m-%d") for t in bench_r.index],
                "daily_ret": bench_r.to_numpy(dtype=float),
                "equity": (1.0 + bench_r).cumprod().to_numpy(dtype=float),
            }
        )
    else:
        bench_df = pd.DataFrame(columns=["date", "daily_ret", "equity"])

    bp = root / "benchmark_equal_weight_equity.csv"
    bench_df.to_csv(bp, index=False)
    paths["benchmark_equal_weight_equity_csv"] = str(bp)

    if trade is not None and trade.equity_curve:
        sp = root / "trader_best_equity.csv"
        pd.DataFrame(trade.equity_curve).to_csv(sp, index=False)
        paths["trader_best_equity_csv"] = str(sp)

    sp_json = root / "summary.json"
    with sp_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False, default=str)
    paths["summary_json"] = str(sp_json)

    png_path = root / "equity_curves.png"
    if _try_save_equity_png(png_path, bench_r, trade):
        paths["equity_png"] = str(png_path)

    return paths


def _try_save_equity_png(path: Path, bench_r: pd.Series, trade: TraderResult | None) -> bool:
    if (bench_r is None or bench_r.empty) and (trade is None or not trade.equity_curve):
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10, 4))
    if bench_r is not None and not bench_r.empty:
        eq = (1.0 + bench_r).cumprod()
        ax.plot(eq.index, eq.values, label="equal_weight_benchmark", linewidth=1.1)
    if trade is not None and trade.equity_curve:
        df = pd.DataFrame(trade.equity_curve)
        if not df.empty:
            d = pd.to_datetime(df["date"])
            ax.plot(d, df["equity"].astype(float), label="trader_search_best", linewidth=1.1)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    ax.set_ylabel("Equity (start=1)")
    ax.set_title("AlphaCrafter — equity curves")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
