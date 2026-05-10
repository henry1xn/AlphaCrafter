"""Create SQLite file and apply shared-memory schema (H)."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root (parent of scripts/) so `python scripts\init_shared_memory.py` works without `pip install -e .`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alphacrafter.memory import init_database


def main() -> int:
    path = init_database()
    print(f"Initialized shared memory at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
