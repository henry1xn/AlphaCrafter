-- AlphaCrafter shared memory H (SQLite)
-- Phase 1: core tables for factors, regimes, ensembles, strategies, executions.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factor_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    ic REAL,
    ir REAL,
    outcome_meta TEXT NOT NULL,
    in_library INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_factor_records_hash ON factor_records (code_hash);
CREATE INDEX IF NOT EXISTS idx_factor_records_library ON factor_records (in_library);

CREATE TABLE IF NOT EXISTS market_regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    features_json TEXT,
    raw_assessment TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ensembles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_id INTEGER,
    members_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (regime_id) REFERENCES market_regimes (id)
);

CREATE TABLE IF NOT EXISTS strategy_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ensemble_id INTEGER,
    spec_json TEXT NOT NULL,
    backtest_score REAL,
    backtest_metrics_json TEXT,
    exploration_meta TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ensemble_id) REFERENCES ensembles (id)
);

CREATE TABLE IF NOT EXISTS strategy_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_candidate_id INTEGER,
    result_json TEXT,
    execution_meta TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (strategy_candidate_id) REFERENCES strategy_candidates (id)
);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    as_of_note TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
