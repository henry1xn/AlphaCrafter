"""Sandboxed execution of LLM-produced factor code on a long ``panel``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
}


def execute_factor_code(code: str, panel: pd.DataFrame) -> pd.Series:
    """
    Execute ``code`` with ``panel`` in namespace; expects variable ``factor`` (Series or ndarray).

    Raises on unsafe operations or type mismatch.
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("panel must be a non-empty DataFrame")

    ns: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "panel": panel,
    }
    try:
        exec(compile(code, "<factor>", "exec"), ns, ns)
    except Exception as exc:  # noqa: BLE001 — surfaced to Miner
        raise RuntimeError(f"exec failed: {exc}") from exc

    factor = ns.get("factor")
    if factor is None:
        raise RuntimeError("factor variable not defined after exec")
    if isinstance(factor, np.ndarray):
        factor = pd.Series(factor, index=panel.index)
    if not isinstance(factor, pd.Series):
        raise TypeError(f"factor must be Series or ndarray, got {type(factor)}")
    if len(factor) != len(panel):
        raise ValueError(f"factor length {len(factor)} != panel length {len(panel)}")
    out = pd.to_numeric(factor, errors="coerce").astype("float64")
    out.index = panel.index
    return out
