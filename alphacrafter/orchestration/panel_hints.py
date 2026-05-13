"""Lightweight panel sanity hints for JSON summaries (no hard failures)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from alphacrafter.data.splits import paper_split_range


def training_panel_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    """
    Calendar coverage hints for Miner IC stability (especially crypto with late start dates).

    Emits ``warnings`` string codes only; callers attach to pipeline JSON.
    """
    if panel.empty or "date" not in panel.columns:
        return {"empty": True, "warnings": ["empty_panel"]}
    d = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    d = d.dropna()
    if d.empty:
        return {"empty": True, "warnings": ["no_parseable_dates"]}
    u = int(d.nunique())
    tmin, tmax = d.min(), d.max()
    span = int((tmax - tmin).days) + 1
    t0, t1 = paper_split_range("training")
    warnings: list[str] = []
    if u < 252:
        warnings.append("unique_dates_lt_252_ic_noisy")
    if u < 400:
        warnings.append("unique_dates_lt_400_consider_longer_history")
    gap_days = (tmin.date() - t0).days
    if gap_days > 45:
        warnings.append("panel_starts_over_45d_after_table1_training_begin")
    if tmin.date() > date(2020, 1, 1):
        warnings.append("training_min_date_after_2020_limited_regimes_for_ic")
    return {
        "empty": False,
        "unique_dates": u,
        "calendar_span_days": span,
        "min_date": tmin.date().isoformat(),
        "max_date": tmax.date().isoformat(),
        "table1_training_inclusive": {"start": t0.isoformat(), "end": t1.isoformat()},
        "warnings": warnings,
    }
