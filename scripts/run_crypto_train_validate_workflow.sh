#!/usr/bin/env bash
# 仅因子实验：训练窗 Miner 挖因子（不写 Screener/Trader）→ IC/IR 报告与因子库快照 → 诊断图。
# 不做策略回测；样本内外只看因子 IC、IR。
#
# Usage:
#   bash scripts/run_crypto_train_validate_workflow.sh /path/to/klines [tickers] [artifact_parent]
# Or: export ALPHACRAFTER_CRYPTO_DATA_DIR=/path/to/klines && bash scripts/run_crypto_train_validate_workflow.sh
#
# 建议本实验独占库文件（避免历史 factor_records 混入）：
#   export ALPHACRAFTER_DB_PATH="${ART}/experiment.sqlite"
# 脚本内已按 ART 自动设置。

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
ART="${3:-runs/factor_ic_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${ART}"

export ALPHACRAFTER_CRYPTO_DATA_DIR="${CRYPTO_DIR}"
export ALPHACRAFTER_DB_PATH="${ART}/experiment.sqlite"

echo "== Step 1: training window, Miner only (populate Z; no Screener/Trader) =="
python scripts/run_alphacrafter.py \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --split training \
  --miner-only \
  --tickers "${TICKERS}" \
  --crypto-rank-by volume \
  --artifacts-dir "${ART}/step1_miner_training"

echo "== Step 2: factor IC/IR report + library snapshot + Markdown (train vs validation) =="
python scripts/validate_experiment.py \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --tickers "${TICKERS}" \
  --train-split training \
  --oos-split validation \
  --json-out "${ART}/factor_ic_ir_report.json" \
  --csv-out "${ART}/factor_ic_ir_table.csv" \
  --markdown-out "${ART}/factor_ic_ir_report.md" \
  --library-json-out "${ART}/factor_library_snapshot.json"

echo "== Step 3: diagnostic plots (IC/IR vs attempts; train vs OOS IC & IR) =="
python scripts/plot_diagnostics.py \
  --db "${ALPHACRAFTER_DB_PATH}" \
  --out-dir "${ART}/plots" \
  --tail 800 \
  --crypto-data-dir "${CRYPTO_DIR}" \
  --tickers "${TICKERS}" \
  --crypto-rank-by volume \
  --oos-split validation \
  --skip-portfolio-sharpe-plot

echo "Done. Artifacts under: ${ART}"
echo "  DB: ${ALPHACRAFTER_DB_PATH}"
