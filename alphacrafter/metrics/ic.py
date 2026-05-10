"""Cross-sectional IC / IR on a long-format panel."""

from __future__ import annotations

import warnings

import pandas as pd


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho without SciPy; avoids degenerate Pearson (constant ranks / zero std)."""
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    if len(rx) < 3:
        return float("nan")
    sdx = float(rx.std(ddof=1))
    sdy = float(ry.std(ddof=1))
    if sdx < 1e-15 or sdy < 1e-15 or not (sdx == sdx) or not (sdy == sdy):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = rx.corr(ry, method="pearson")
    return float(c) if c == c else float("nan")


def cross_sectional_ic_ir(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    factor_col: str = "factor",
    fwd_ret_col: str = "fwd_ret",
    min_names: int = 8,
) -> tuple[float, float]:
    """
    Mean Spearman IC across dates and IR = mean(IC) / std(IC).

    Returns (nan, nan) if insufficient data.
    """
    need = {date_col, factor_col, fwd_ret_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"cross_sectional_ic_ir missing columns: {missing}")

    work = df[[date_col, factor_col, fwd_ret_col]].dropna()
    ics: list[float] = []
    for _, g in work.groupby(date_col, sort=False):
        if len(g) < min_names:
            continue
        ic = _spearman_corr(g[factor_col], g[fwd_ret_col])
        if ic == ic:  # not NaN
            ics.append(float(ic))
    if len(ics) < 3:
        return float("nan"), float("nan")
    s = pd.Series(ics, dtype="float64")
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ir = m / sd if sd > 1e-12 else m
    return m, float(ir)
