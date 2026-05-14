"""Train vs OOS validation for crypto: factor IC/IR, optional LS Sharpe, benchmark context."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alphacrafter.agents.factor_exec import execute_factor_code
from alphacrafter.agents.miner import MinerAgent
from alphacrafter.backtest.vectorized import backtest_long_short, daily_portfolio_metrics, pivot_close_returns
from alphacrafter.data.panel import add_forward_return, build_long_panel_crypto
from alphacrafter.data.splits import SplitName, normalize_split_name, paper_split_range


def _panel_without_fwd(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.drop(columns=["fwd_ret"], errors="ignore").copy()


def benchmark_equal_weight_metrics(panel: pd.DataFrame) -> dict[str, float]:
    """Equal-weight long-only daily returns (same definition as orchestration benchmark)."""
    ret_w = pivot_close_returns(panel)
    if ret_w.empty:
        return daily_portfolio_metrics(pd.Series(dtype=float))
    bench_r = ret_w.mean(axis=1).dropna()
    return daily_portfolio_metrics(bench_r)


def factor_long_short_metrics(panel: pd.DataFrame, code: str) -> dict[str, float | None]:
    """
    Cross-sectional dollar-neutral backtest from raw factor values (wide z inside backtest_long_short).
    """
    base = _panel_without_fwd(panel)
    try:
        fac = execute_factor_code(code, base)
        df = panel[["date", "ticker"]].copy()
        df["f"] = fac.values
        wide_sig = df.pivot(index="date", columns="ticker", values="f").sort_index()
        ret = pivot_close_returns(panel)
        _, m = backtest_long_short(wide_sig, ret, signal_lag=1)
        return {k: (float(v) if v is not None and v == v else None) for k, v in m.items()}  # type: ignore[misc]
    except Exception:
        return {
            "sharpe_ann": None,
            "mean_daily": None,
            "cum_return": None,
            "n": None,
            "ann_return_pct": None,
            "max_drawdown_pct": None,
        }


def build_panels_for_paper_splits(
    tickers: list[str],
    crypto_dir: str | Path,
    *,
    train_split: SplitName,
    oos_split: SplitName,
    trading_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load long panels with ``fwd_ret`` for two calendar phases (Table 1 style)."""
    t0, t1 = paper_split_range(train_split)
    v0, v1 = paper_split_range(oos_split)
    train = build_long_panel_crypto(
        tickers, crypto_dir, start=t0, end=t1, trading_days=int(trading_days)
    )
    oos = build_long_panel_crypto(
        tickers, crypto_dir, start=v0, end=v1, trading_days=int(trading_days)
    )
    return add_forward_return(train), add_forward_return(oos)


def evaluate_library_train_oos(
    *,
    library_rows: Iterable[Any],
    miner: MinerAgent,
    train_panel: pd.DataFrame,
    oos_panel: pd.DataFrame,
    include_sharpe: bool,
) -> dict[str, Any]:
    """
    For each active Z row: IC/IR on train and on OOS; optional long/short Sharpe on both windows.

    ``library_rows`` should be rows from ``SharedMemory.list_library_factors()`` (sqlite3.Row-like).
    """
    factors: list[dict[str, Any]] = []
    ic_tr_list: list[float] = []
    ic_oos_list: list[float] = []

    for row in library_rows:
        rid = int(row["id"])
        code = str(row["code"])
        h = str(row["code_hash"])
        ic_tr, ir_tr, etr = miner.validate(code, train_panel)
        ic_oos, ir_oos, eoos = miner.validate(code, oos_panel)

        rec: dict[str, Any] = {
            "factor_record_id": rid,
            "code_hash": h,
            "stored_ic": float(row["ic"]) if row["ic"] is not None and row["ic"] == row["ic"] else None,
            "stored_ir": float(row["ir"]) if row["ir"] is not None and row["ir"] == row["ir"] else None,
            "ic_train": float(ic_tr) if ic_tr is not None and ic_tr == ic_tr else None,
            "ir_train": float(ir_tr) if ir_tr is not None and ir_tr == ir_tr else None,
            "ic_oos": float(ic_oos) if ic_oos is not None and ic_oos == ic_oos else None,
            "ir_oos": float(ir_oos) if ir_oos is not None and ir_oos == ir_oos else None,
            "err_train": etr,
            "err_oos": eoos,
        }
        if include_sharpe:
            prev = os.environ.get("ALPHACRAFTER_ASSET_CLASS")
            try:
                os.environ["ALPHACRAFTER_ASSET_CLASS"] = "crypto"
                rec["ls_sharpe_train"] = factor_long_short_metrics(train_panel, code).get("sharpe_ann")
                rec["ls_sharpe_oos"] = factor_long_short_metrics(oos_panel, code).get("sharpe_ann")
            finally:
                if prev is None:
                    os.environ.pop("ALPHACRAFTER_ASSET_CLASS", None)
                else:
                    os.environ["ALPHACRAFTER_ASSET_CLASS"] = prev

        factors.append(rec)

        if rec["ic_train"] is not None and rec["ic_oos"] is not None:
            ic_tr_list.append(float(rec["ic_train"]))
            ic_oos_list.append(float(rec["ic_oos"]))

    summary: dict[str, Any] = {
        "n_library_factors": len(factors),
        "n_both_ic_finite": len(ic_tr_list),
    }
    if ic_tr_list:
        summary["mean_ic_train"] = float(np.mean(ic_tr_list))
        summary["mean_ic_oos"] = float(np.mean(ic_oos_list))
        summary["median_ic_train"] = float(np.median(ic_tr_list))
        summary["median_ic_oos"] = float(np.median(ic_oos_list))
        if len(ic_tr_list) >= 2:
            c = float(np.corrcoef(ic_tr_list, ic_oos_list)[0, 1])
            summary["corr_ic_train_vs_oos"] = c if c == c else None
        else:
            summary["corr_ic_train_vs_oos"] = None
        summary["ic_oos_positive_share"] = float(sum(1 for x in ic_oos_list if x > 0) / len(ic_oos_list))
    else:
        summary["mean_ic_train"] = None
        summary["mean_ic_oos"] = None
        summary["median_ic_train"] = None
        summary["median_ic_oos"] = None
        summary["corr_ic_train_vs_oos"] = None
        summary["ic_oos_positive_share"] = None

    return {"factors": factors, "summary": summary}


def run_crypto_validation_report(
    *,
    crypto_dir: Path,
    tickers: list[str],
    library_rows: list[Any],
    miner: MinerAgent,
    train_split: str = "training",
    oos_split: str = "validation",
    trading_days: int = 200,
    include_sharpe: bool = False,
    include_benchmark: bool = False,
) -> dict[str, Any]:
    """
    High-level report: per-factor train/OOS IC/IR; optional LS Sharpe; optional equal-weight benchmark.

    ``train_split`` / ``oos_split`` are normalized like ``--split`` in the CLI (Table 1 calendar).
    """
    ts = normalize_split_name(train_split)
    os_ = normalize_split_name(oos_split)
    if ts is None or os_ is None:
        raise ValueError("train_split and oos_split must be valid paper split names")

    train_panel, oos_panel = build_panels_for_paper_splits(
        tickers, crypto_dir, train_split=ts, oos_split=os_, trading_days=int(trading_days)
    )

    bench_tr = benchmark_equal_weight_metrics(train_panel) if include_benchmark else None
    bench_oos = benchmark_equal_weight_metrics(oos_panel) if include_benchmark else None

    if train_panel.empty or oos_panel.empty:
        err: dict[str, Any] = {
            "ok": False,
            "error": "empty_train_or_oos_panel",
            "train_split": ts,
            "oos_split": os_,
            "crypto_data_dir": str(crypto_dir.resolve()),
            "tickers_used": tickers,
            "panel_rows": {"train": int(len(train_panel)), "oos": int(len(oos_panel))},
            "unique_dates": {
                "train": int(pd.to_datetime(train_panel["date"]).dt.normalize().nunique())
                if not train_panel.empty
                else 0,
                "oos": int(pd.to_datetime(oos_panel["date"]).dt.normalize().nunique())
                if not oos_panel.empty
                else 0,
            },
            "factor_validation": {"factors": [], "summary": {"n_library_factors": len(library_rows)}},
        }
        if include_benchmark and bench_tr is not None and bench_oos is not None:
            err["benchmark_equal_weight"] = {"train": bench_tr, "oos": bench_oos}
        return err

    fac_block = evaluate_library_train_oos(
        library_rows=library_rows,
        miner=miner,
        train_panel=train_panel,
        oos_panel=oos_panel,
        include_sharpe=include_sharpe,
    )

    out: dict[str, Any] = {
        "ok": True,
        "train_split": ts,
        "oos_split": os_,
        "crypto_data_dir": str(crypto_dir.resolve()),
        "tickers_used": tickers,
        "panel_rows": {"train": int(len(train_panel)), "oos": int(len(oos_panel))},
        "unique_dates": {
            "train": int(pd.to_datetime(train_panel["date"]).dt.normalize().nunique()),
            "oos": int(pd.to_datetime(oos_panel["date"]).dt.normalize().nunique()),
        },
        "factor_validation": fac_block,
    }
    if include_benchmark and bench_tr is not None and bench_oos is not None:
        out["benchmark_equal_weight"] = {"train": bench_tr, "oos": bench_oos}
    return out


def ic_sharpe_table_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """Flatten factor_validation.factors into a DataFrame for CSV export."""
    rows = (report.get("factor_validation") or {}).get("factors") or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def factor_library_rows_to_jsonable(library_rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Snapshot active Z for a JSON file (includes full factor code)."""
    out: list[dict[str, Any]] = []
    for row in library_rows:
        ic = row["ic"]
        ir = row["ir"]
        out.append(
            {
                "factor_record_id": int(row["id"]),
                "code_hash": str(row["code_hash"]),
                "stored_ic": float(ic) if ic is not None and ic == ic else None,
                "stored_ir": float(ir) if ir is not None and ir == ir else None,
                "outcome_meta": str(row["outcome_meta"]) if row["outcome_meta"] is not None else None,
                "created_at": str(row["created_at"]) if row["created_at"] is not None else None,
                "code": str(row["code"]),
            }
        )
    return out


def write_factor_validation_markdown(report: dict[str, Any], path: Path) -> None:
    """Short human-readable summary for 样本内/外 IC·IR 分析."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 因子验证报告（IC / IR）\n")
    lines.append(f"- 训练窗: `{report.get('train_split')}`\n")
    lines.append(f"- 样本外窗: `{report.get('oos_split')}`\n")
    lines.append(f"- 数据目录: `{report.get('crypto_data_dir')}`\n")
    lines.append(f"- 标的数: {len(report.get('tickers_used') or [])}\n")
    ud = report.get("unique_dates") or {}
    lines.append(f"- 训练窗交易日数（面板）: {ud.get('train')}\n")
    lines.append(f"- 样本外窗交易日数（面板）: {ud.get('oos')}\n")
    lines.append(f"- 状态: **{'ok' if report.get('ok') else '失败'}**")
    if report.get("error"):
        lines.append(f" (`{report['error']}`)")
    lines.append("\n\n## 汇总（因子库 Z）\n\n")
    summ = (report.get("factor_validation") or {}).get("summary") or {}
    for k in sorted(summ.keys()):
        lines.append(f"- **{k}**: `{summ[k]}`\n")
    lines.append("\n## 每个因子（复算 IC/IR）\n\n")
    lines.append("| id | hash(前12) | IC训练 | IR训练 | IC样本外 | IR样本外 |\n")
    lines.append("|---:|---|---:|---:|---:|---:|\n")
    for f in (report.get("factor_validation") or {}).get("factors") or []:
        hid = str(f.get("code_hash", ""))[:12]
        lines.append(
            f"| {f.get('factor_record_id')} | `{hid}`… | "
            f"{_fmt_num(f.get('ic_train'))} | {_fmt_num(f.get('ir_train'))} | "
            f"{_fmt_num(f.get('ic_oos'))} | {_fmt_num(f.get('ir_oos'))} |\n"
        )
    lines.append(
        "\n> 说明：`stored_ic`/`stored_ir` 为入库时训练窗上的记录；表中为本次在训练/样本外窗上**重新计算**的 IC/IR。\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def _fmt_num(x: Any) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return ""
    return f"{v:.6f}"
