"""SQLite-backed shared memory H."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from alphacrafter.config.settings import DB_PATH

_SCHEMA_VERSION = "1"


def _schema_sql_path() -> Path:
    return Path(__file__).with_name("schema.sql")


def init_database(db_path: Path | None = None) -> Path:
    """
    Create parent dirs, apply schema, stamp version. Idempotent.
    Returns resolved database path.
    """
    path = Path(db_path or DB_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sm = SharedMemory(path)
    try:
        sm.ensure_schema()
        sm.set_meta("schema_version", _SCHEMA_VERSION)
    finally:
        sm.close()
    return path


class SharedMemory:
    """Thin DAO over H; agents extend usage in later phases."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or DB_PATH).resolve()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SharedMemory":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ensure_schema(self) -> None:
        sql = _schema_sql_path().read_text(encoding="utf-8")
        conn = self.connect()
        conn.executescript(sql)
        conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def get_meta(self, key: str) -> str | None:
        conn = self.connect()
        cur = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return str(row["value"]) if row else None

    @staticmethod
    def hash_code(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def record_factor_event(
        self,
        code: str,
        ic: float | None,
        ir: float | None,
        outcome_meta: str,
        *,
        in_library: bool = False,
    ) -> int:
        """Append a factor validation / maintenance row."""
        conn = self.connect()
        h = self.hash_code(code)
        cur = conn.execute(
            """
            INSERT INTO factor_records (code, code_hash, ic, ir, outcome_meta, in_library)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (code, h, ic, ir, outcome_meta, int(bool(in_library))),
        )
        conn.commit()
        return int(cur.lastrowid)

    def set_factor_library_flag(self, code_hash: str, in_library: int, outcome_meta: str) -> None:
        conn = self.connect()
        conn.execute(
            """
            UPDATE factor_records
            SET in_library = ?, outcome_meta = ?, updated_at = datetime('now')
            WHERE id = (
                SELECT id FROM factor_records WHERE code_hash = ?
                ORDER BY id DESC LIMIT 1
            )
            """,
            (int(in_library), outcome_meta, code_hash),
        )
        conn.commit()

    def list_library_factors(self) -> list[sqlite3.Row]:
        """Active Z: latest row per code_hash marked in_library=1."""
        conn = self.connect()
        cur = conn.execute(
            """
            SELECT fr.*
            FROM factor_records fr
            INNER JOIN (
                SELECT code_hash, MAX(id) AS max_id
                FROM factor_records
                WHERE in_library = 1
                GROUP BY code_hash
            ) t ON fr.id = t.max_id
            ORDER BY fr.id ASC
            """
        )
        return list(cur.fetchall())

    def insert_universe_snapshot(self, source_path: str, row_count: int, as_of_note: str = "") -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO universe_snapshots (source_path, row_count, as_of_note)
            VALUES (?, ?, ?)
            """,
            (source_path, row_count, as_of_note),
        )
        conn.commit()
        return int(cur.lastrowid)

    def insert_market_regime(
        self,
        label: str,
        features: dict[str, Any] | None = None,
        raw_assessment: str | None = None,
    ) -> int:
        conn = self.connect()
        fj = json.dumps(features or {}, ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO market_regimes (label, features_json, raw_assessment)
            VALUES (?, ?, ?)
            """,
            (label, fj, raw_assessment),
        )
        conn.commit()
        return int(cur.lastrowid)

    def insert_ensemble(self, regime_id: int | None, members: Iterable[dict[str, Any]]) -> int:
        conn = self.connect()
        payload = json.dumps(list(members), ensure_ascii=False)
        cur = conn.execute(
            "INSERT INTO ensembles (regime_id, members_json) VALUES (?, ?)",
            (regime_id, payload),
        )
        conn.commit()
        return int(cur.lastrowid)

    def deactivate_factor_library_rows(self, code_hash: str) -> None:
        """Clear in_library for active rows of this factor (audit uses a new insert)."""
        conn = self.connect()
        conn.execute(
            """
            UPDATE factor_records
            SET in_library = 0, updated_at = datetime('now')
            WHERE code_hash = ? AND in_library = 1
            """,
            (code_hash,),
        )
        conn.commit()

    def recent_factor_events(self, limit: int = 30) -> list[sqlite3.Row]:
        conn = self.connect()
        cur = conn.execute(
            """
            SELECT id, code_hash, ic, ir, outcome_meta, in_library, created_at
            FROM factor_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return list(cur.fetchall())

    def insert_strategy_candidate(
        self,
        ensemble_id: int | None,
        spec_json: str,
        backtest_score: float | None,
        backtest_metrics_json: str | None,
        exploration_meta: str,
    ) -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO strategy_candidates
            (ensemble_id, spec_json, backtest_score, backtest_metrics_json, exploration_meta)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ensemble_id, spec_json, backtest_score, backtest_metrics_json, exploration_meta),
        )
        conn.commit()
        return int(cur.lastrowid)

    def insert_strategy_execution(
        self,
        strategy_candidate_id: int | None,
        result_json: str,
        execution_meta: str,
    ) -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO strategy_executions (strategy_candidate_id, result_json, execution_meta)
            VALUES (?, ?, ?)
            """,
            (strategy_candidate_id, result_json, execution_meta),
        )
        conn.commit()
        return int(cur.lastrowid)

    def get_factor_row(self, row_id: int) -> sqlite3.Row | None:
        conn = self.connect()
        cur = conn.execute("SELECT * FROM factor_records WHERE id = ?", (row_id,))
        row = cur.fetchone()
        return row
