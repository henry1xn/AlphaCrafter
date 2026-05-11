#!/usr/bin/env bash
# AlphaCrafter — Linux 纯净环境一键安装（腾讯云 / Ubuntu 等）
# 用法：在项目根目录执行  bash scripts/setup_linux.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "未找到 $PY，请先安装 Python 3.10+（例如: sudo apt install python3 python3-venv）"
  exit 1
fi

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .

if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "[提示] 已从 .env.example 创建 .env，请编辑填入 OPENAI_API_KEY（DeepSeek）等。"
fi

echo ""
echo "安装完成。下一步示例："
echo "  source .venv/bin/activate"
echo "  python scripts/run_alphacrafter.py --crypto-data-dir /path/to/klines --split training --tickers 20"
echo "论文 Table 1 回测段（本地 K 线）："
echo "  python scripts/run_alphacrafter.py --crypto-data-dir /path/to/klines --split backtesting --tickers 20"
echo "（可选）美股维基 + Yahoo："
echo "  python scripts/scrape_sp500.py --out data/raw/sp500_wiki.csv"
echo "  python scripts/run_alphacrafter.py --universe data/raw/sp500_wiki.csv --tickers 20 --days 200 --sleep-panel 0.35"
