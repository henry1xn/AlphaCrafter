#!/usr/bin/env bash
# Paper-style crypto: training (fill Z) → backtesting (eval) → diagnostic plots.
# Usage:
#   bash scripts/run_crypto_paper_workflow.sh /path/to/klines [tickers] [artifact_parent]
# Or:  export ALPHACRAFTER_CRYPTO_DATA_DIR=/path/to/klines
#       bash scripts/run_crypto_paper_workflow.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CRYPTO_DIR="${1:-${ALPHACRAFTER_CRYPTO_DATA_DIR:-}}"
if [[ -z "${CRYPTO_DIR}" ]]; then
  echo "Usage: $0 <crypto_klines_dir> [tickers] [artifact_dir]" >&2
  echo "   or: export ALPHACRAFTER_CRYPTO_DATA_DIR=...  &&  $0" >&2
  exit 1
fi

TICKERS="${2:-20}"
ART="${3:-runs/workflow_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${ART}"

export ALPHACRAFTER_CRYPTO_DATA_DIR="${CRYPTO_DIR}"

echo "== Step 1: training (Miner on training window, populate Z) =="
python scripts/run_alphacrafter.py \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --split training \
  --tickers "${TICKERS}" \
  --crypto-rank-by volume \
  --artifacts-dir "${ART}/step1_training"

echo "== Step 2: backtesting (Miner read-only on Z; Screener+Trader on backtest window) =="
python scripts/run_alphacrafter.py \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --split backtesting \
  --tickers "${TICKERS}" \
  --crypto-rank-by volume \
  --artifacts-dir "${ART}/step2_backtesting"

echo "== Step 3: diagnostic plots (SQLite H) =="
python scripts/plot_diagnostics.py \
  --out-dir "${ART}/plots" \
  --tail 800 \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --tickers "${TICKERS}" \
  --crypto-rank-by volume \
  --oos-split backtesting

echo "Done. Artifacts: ${ART}"
