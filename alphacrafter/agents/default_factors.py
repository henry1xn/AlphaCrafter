"""Curated factor snippets used when the Miner leaves Z empty (demo / short windows)."""

from __future__ import annotations

# Same execution rules as LLM output: no import; np/pd injected by sandbox.
BUILTIN_FACTOR_CODES: list[str] = [
    (
        "g = panel.groupby('ticker')['close']\n"
        "mom = g.pct_change(20)\n"
        "factor = mom.replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
    ),
    (
        "g = panel.groupby('ticker')['close']\n"
        "rv = g.pct_change().rolling(10, min_periods=5).std()\n"
        "factor = rv.replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
    ),
    (
        "rev = -panel.groupby('ticker')['close'].pct_change(1)\n"
        "factor = rev.replace([np.inf, -np.inf], np.nan).fillna(0.0)\n"
    ),
    (
        "rk = panel.groupby('date')['volume'].rank(pct=True)\n"
        "factor = (0.5 - rk).fillna(0.0)\n"
    ),
]
