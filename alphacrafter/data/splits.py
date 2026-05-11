"""Paper Table 1 style chronological splits (Training / Validation / Backtesting / Live).

Boundaries follow the notation YYYY.MM in the paper: each phase is a contiguous
calendar span on daily bars. CSI 300 uses the **same calendar cut points** as
S&P 500 in Table 1; only the reported point counts differ (holidays / universe).

We use **inclusive** calendar ``start`` / ``end`` dates for filtering panels.
For Yahoo ``period2`` (typically exclusive end instant), callers add one day
after the inclusive end when fetching.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import pandas as pd

SplitName = Literal["training", "validation", "backtesting", "live_trading"]

# Phases where the factor library Z must not be updated — train Miner on ``training`` only, then evaluate here.
EVAL_SPLITS: frozenset[SplitName] = frozenset({"validation", "backtesting", "live_trading"})

# Inclusive last calendar day per phase (aligned with Table 1 month boundaries).
# Training ends before 2023-01; Validation is calendar 2023; Backtest is 2024–2025;
# Live is Jan–Apr 2026.
_SP500_INCLUSIVE: dict[SplitName, tuple[date, date]] = {
    "training": (date(2016, 1, 1), date(2022, 12, 31)),
    "validation": (date(2023, 1, 1), date(2023, 12, 31)),
    "backtesting": (date(2024, 1, 1), date(2025, 12, 31)),
    "live_trading": (date(2026, 1, 1), date(2026, 4, 30)),
}

# Table 1 reported counts — for metadata / sanity checks only (actual bars vary by venue).
EXPECTED_TRADING_DAYS_SP500: dict[SplitName, int] = {
    "training": 1763,
    "validation": 250,
    "backtesting": 502,
    "live_trading": 61,
}
EXPECTED_TRADING_DAYS_CSI300: dict[SplitName, int] = {
    "training": 1703,
    "validation": 242,
    "backtesting": 486,
    "live_trading": 56,
}


def normalize_split_name(name: str | None) -> SplitName | None:
    if not name or not str(name).strip():
        return None
    key = str(name).strip().lower().replace("-", "_")
    aliases = {"live": "live_trading", "bt": "backtesting", "val": "validation", "train": "training"}
    key = aliases.get(key, key)
    if key in _SP500_INCLUSIVE:
        return key  # type: ignore[return-value]
    raise ValueError(
        f"Unknown dataset split {name!r}. "
        f"Use one of: {', '.join(sorted(_SP500_INCLUSIVE.keys()))} "
        "(aliases: train, val, bt, live)."
    )


def paper_split_range(split: SplitName | str) -> tuple[date, date]:
    """Return inclusive (start, end) calendar dates for the paper phase."""
    if isinstance(split, str):
        sn = normalize_split_name(split)
    elif split in _SP500_INCLUSIVE:
        sn = split
    else:
        sn = normalize_split_name(str(split))
    if sn is None:
        raise ValueError("split is required")
    return _SP500_INCLUSIVE[sn]


def yahoo_period2_exclusive_end(inclusive_end: date) -> date:
    """Yahoo chart ``period2`` behaves as an exclusive upper bound — use day after last bar."""
    return inclusive_end + timedelta(days=1)


def filter_panel_to_date_range(
    panel: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Keep rows with date in [start, end] inclusive (date column normalized to midnight)."""
    if panel.empty:
        return panel
    d = pd.to_datetime(panel["date"]).dt.normalize()
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    out = panel.loc[(d >= lo) & (d <= hi)].copy()
    return out.reset_index(drop=True)


def count_unique_trading_dates(panel: pd.DataFrame) -> int:
    if panel.empty or "date" not in panel.columns:
        return 0
    return int(pd.to_datetime(panel["date"]).dt.normalize().nunique())


def split_metadata(split: SplitName) -> dict[str, object]:
    start, end = paper_split_range(split)
    return {
        "split": split,
        "calendar": "paper_table1_training_validation_backtest_live",
        "start_inclusive": start.isoformat(),
        "end_inclusive": end.isoformat(),
        "expected_trading_days_sp500_table1": EXPECTED_TRADING_DAYS_SP500[split],
        "expected_trading_days_csi300_table1": EXPECTED_TRADING_DAYS_CSI300[split],
    }
