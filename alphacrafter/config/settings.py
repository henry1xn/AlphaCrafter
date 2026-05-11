"""Paths and environment-driven settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = parent of package `alphacrafter/`
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_data_env = os.getenv("ALPHACRAFTER_DATA_DIR", "").strip()
DATA_DIR = Path(_data_env) if _data_env else (PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"

_db_env = os.getenv("ALPHACRAFTER_DB_PATH", "").strip()
DB_PATH = Path(_db_env) if _db_env else (DATA_DIR / "shared_memory.db")

HTTP_SLEEP_SEC = float(os.getenv("ALPHACRAFTER_HTTP_SLEEP", "0.35"))

# --- Orchestration / agents ---
ORCH_TICKER_LIMIT = int(os.getenv("ALPHACRAFTER_ORCH_TICKER_LIMIT", "20"))
MINER_MAX_ITERATIONS = int(os.getenv("ALPHACRAFTER_MINER_MAX_ITERATIONS", "4"))
MINER_IC_ACCEPT = float(os.getenv("ALPHACRAFTER_MINER_IC_ACCEPT", "0.03"))
MINER_IC_RETAIN = float(os.getenv("ALPHACRAFTER_MINER_IC_RETAIN", "0.01"))
SCREENER_MIN_FACTORS = int(os.getenv("ALPHACRAFTER_SCREENER_MIN_FACTORS", "1"))
SCREENER_TOP_K = int(os.getenv("ALPHACRAFTER_SCREENER_TOP_K", "10"))
TRADER_MAX_EXPLORATIONS = int(os.getenv("ALPHACRAFTER_TRADER_MAX_EXPLORATIONS", "8"))
PANEL_TRADING_DAYS = int(os.getenv("ALPHACRAFTER_PANEL_DAYS", "200"))

for _p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)
