"""``python -m alphacrafter`` — runs a single orchestration pass."""

from __future__ import annotations

import json
import sys

from alphacrafter.orchestration.loop import run_pipeline


def main() -> int:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
