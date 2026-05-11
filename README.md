# AlphaCrafter

基于多智能体（Miner / Screener / Trader）与共享内存（SQLite）的量化研究与交易框架复现。

**默认数据路径（推荐）：** 在本地或服务器目录中放置 **按交易对分文件的加密货币 K 线**（`*.csv` / `*.parquet`），通过 `--crypto-data-dir` 或环境变量 `ALPHACRAFTER_CRYPTO_DATA_DIR` 指向该目录即可运行，**无需 Yahoo、无需维基 S&P500 列表**。  
可选遗留路径：仍支持 `--universe` + Yahoo 拉取美股日线（需 `requests` 访问 Yahoo）。

---

## 加密货币 K 线：目录与列名

在**同一目录**下放多个文件，文件名即标的，例如：`BTCUSDT.csv`、`ETHUSDT.parquet`。  
支持的列名（不区分大小写，自动映射）：

- 时间：`open_time` / `timestamp` / `time` / `datetime` / `date` / `ts`（可为 Unix ms/s 或 ISO 字符串）
- OHLCV：`open, high, low, close`（或 `o/h/l/c`），`volume`（或 `quote_volume` 等）

若单标的存在**同一天多根 K 线**（如小时线），加载时会**按日历日聚合为日 OHLCV**（24/7 连续序列，不按美股交易日剔除周末）。

**宇宙截断：** 默认按近 `ALPHACRAFTER_CRYPTO_LOOKBACK_DAYS`（默认 90）日历日内的 **成交量合计** 排序，取 `--tickers` 上限；`--crypto-rank-by none` 则按文件名顺序取前 N。财报市值无法从 OHLCV 推断，故不提供 `marketcap` 排序。

---

## 纯净部署：腾讯云 Linux（加密货币模式）

需能访问你的 **LLM API**；数据为**本机目录**（可将远程盘挂载到该路径）。

### A. Python 虚拟环境

```bash
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/henry1xn/AlphaCrafter.git
cd AlphaCrafter
bash scripts/setup_linux.sh
source .venv/bin/activate
nano .env   # OPENAI_API_KEY、ALPHACRAFTER_LLM_PROVIDER=openai 等；可选 ALPHACRAFTER_CRYPTO_DATA_DIR=/path/to/klines

# 示例：Table 1 回测段 + 本地 K 线（把 /data/crypto 换成你的路径）
python scripts/run_alphacrafter.py \
  --crypto-data-dir /data/crypto \
  --split backtesting \
  --tickers 20 \
  --crypto-rank-by volume
```

### B. Docker（把宿主机的 K 线目录挂进容器）

在 `docker-compose.yml` 里为服务增加 volume，例如 `- /host/crypto:/data/crypto:ro`，然后：

```bash
docker compose run --rm orchestrate python scripts/run_alphacrafter.py \
  --crypto-data-dir /data/crypto \
  --split backtesting \
  --tickers 20
```

### 论文 Table 1 日历分段（与 README 原逻辑一致）

使用 `--split` 指定阶段（**日历日**边界不变；用于过滤本地 K 线的 `date`）：

```bash
python scripts/run_alphacrafter.py \
  --crypto-data-dir /path/to/klines \
  --split backtesting \
  --tickers 20
```

可选值：`training`、`validation`、`backtesting`、`live_trading`（别名：`train`、`val`、`bt`、`live`）。也可在 `.env` 中设置 `ALPHACRAFTER_PAPER_SPLIT`。

#### 因子库 Z：仅在训练窗写入（与论文分段一致）

当 **`--split` 为 `validation` / `backtesting` / `live_trading`** 且数据由本进程从 **本地目录或 Yahoo** 加载时：

1. **先**在 **training** 窗（2016-01-01 — 2022-12-31）上运行 **Miner**（及可选 builtin seed），**仅此窗可向 Z 写入**。  
2. **再**在当前阶段窗上只跑 **Screener → Trader**；eval 窗 **不写 Z**（`miner_seed.reason`：`eval_phase_no_Z_writes`）。  
3. JSON 字段 **`library_discipline`** 标明 `data_source: crypto_local` 或 `yahoo`。

`--split training`：Miner → Screener → Trader 均在训练窗。

**单元测试里注入的 `panel`** 仍为 `injected_panel` 模式，不自动拉训练窗。

---

## 遗留：美股 Yahoo + 维基宇宙（可选）

```bash
python scripts/scrape_sp500.py --out data/raw/sp500_wiki.csv
python scripts/run_alphacrafter.py --universe data/raw/sp500_wiki.csv --tickers 20 --days 200 --sleep-panel 0.35
```

### Docker（美股示例）

```bash
docker compose run --rm scrape-sp500
docker compose run --rm orchestrate \
  python scripts/run_alphacrafter.py \
  --universe /app/data/raw/sp500_wiki.csv \
  --split backtesting \
  --tickers 20 \
  --sleep-panel 0.35
```

### 自检（可选）

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py" -v
```

### 说明（与论文「完全复现」还差什么）

当前已实现：**Table 1 日历切分**、**验证/回测/实盘窗内不向 Z 写入**（训练窗独占 Miner + seed）、以及 **Screener / Trader 在对应窗上的评估**。论文表格中的绝对数值仍依赖 **同款标的全集、更长样本、超参与随机种子** 等；本仓库优先保证 **流程与泄露控制** 合理，数值需在云上按论文规模自行复现实验。

---

## 一、架构总览

### 1.1 符号与组件映射

| 符号 | 含义 | 实现要点 |
|------|------|----------|
| **Z** | 因子库 | 已通过检验的因子公式集合（及元数据） |
| **U** | 资产宇宙 | 本地 K 线文件名推导的交易对列表，或 legacy CSV 宇宙 |
| **H** | 共享内存 | 中央 SQLite：因子代码、IC/IR、市场体制日志、策略与回测/执行结果 |
| **M** | 市场状态 | 波动、趋势等可由行情聚合的指标，供 Screener 体制判断 |
| **E** | 因子组合 | 经筛选、赋权、方向的因子子集 |
| **R̂** | 市场体制 | LLM 对当前波动/趋势等的评估标签 |
| **π** | 策略 | 由 Trader 构造并经回测择优的策略描述与参数 |

### 1.2 数据流（逻辑）

```text
[数据源: 本地 crypto K 线目录 或 S&P500+Yahoo（可选）]
        │
        ▼
   [Data Pipeline] ──写入──► [SQLite: H]
        ▲                        │
        │                        ├── Miner: 生成/校验因子 → 更新 Z 与 H
        │                        ├── Screener: M,H,Z → R̂,E → 写 H
        │                        └── Trader: E,R̂,H → π,回测,（模拟）实盘 → 写 H
        │
[Orchestration 主循环按阶段调度上述智能体]
```

### 1.3 智能体职责（与伪代码对应）

- **Miner**：由 LLM 生成 Pandas 因子并校验 IC/IR（阈值见 `.env`）；维护因子库 **Z**。
- **Screener**：体制评估、因子 suitability、组合成 **E**。
- **Trader**：策略网格搜索、向量回测；`live_trading` 为**模拟占位**（非真实券商）。

### 1.4 技术栈

- Python **3.10+**
- **pandas**, **numpy**, **requests**, **beautifulsoup4**, **lxml**
- **SQLite**（`data/shared_memory.db`）
- **LLM**：Anthropic / OpenAI 兼容（DeepSeek）/ MiniMax / `stub`（见 `alphacrafter/utils/llm.py`）

---

## 二、目录结构（当前）

```text
AlphaCrafter/
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
├── .env.example
├── scripts/
│   ├── setup_linux.sh       # Linux 一键安装
│   ├── scrape_sp500.py
│   └── run_alphacrafter.py  # 编排入口（含 --split）
├── alphacrafter/data/local_klines.py  # 本地 K 线读取
├── data/raw/                # 可选：维基抓取 CSV（默认不提交）
├── tests/
└── README.md
```

---

## 三、Windows 本地（Anaconda / venv）

```powershell
cd e:\project\AlphaCrafter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
python scripts\run_alphacrafter.py --crypto-data-dir D:\crypto_klines --split training --tickers 10 --crypto-rank-by volume
```

---

## 四、环境变量摘要（`.env`）

详见 `.env.example`。常用项：

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | DeepSeek 等 OpenAI 兼容接口密钥 |
| `OPENAI_BASE_URL` | 默认 `https://api.deepseek.com` |
| `OPENAI_MODEL` | 例如 `deepseek-chat` |
| `ALPHACRAFTER_LLM_PROVIDER` | `openai` / `anthropic` / `stub` 等 |
| `ALPHACRAFTER_PAPER_SPLIT` | 可选：`training` / `validation` / `backtesting` / `live_trading` |
| `ALPHACRAFTER_ORCH_TICKER_LIMIT` | 默认单次编排所用标的数量上限 |
| `ALPHACRAFTER_CRYPTO_DATA_DIR` | 与 CLI `--crypto-data-dir` 等价，可二选一 |
| `ALPHACRAFTER_CRYPTO_RANK_BY` | `volume` / `none` |
| `ALPHACRAFTER_BARS_PER_YEAR` | 年化用 bar 数；不设且为 crypto 时默认 **365** |

勿将 `.env` 提交到 Git。

---

## 五、免责声明

本仓库用于研究与教育目的。任何回测结果不构成投资建议；实盘交易需自行承担风险并遵守当地法规与交易所规则。
