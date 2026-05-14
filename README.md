# AlphaCrafter（服务器加密货币回测）

多智能体（**Miner / Screener / Trader**）+ **SQLite 共享内存 H**，在**本地 Parquet/CSV K 线**上做因子挖掘、体制筛选与向量回测。  
**当前文档以你在服务器上的路径为准**，不再展开美股 Yahoo / 维基流程（相关脚本仍保留在仓库中，需要时自行使用）。

---

## 你要做的三件事

1. **配好环境**：Python 依赖 + `.env` 里的 **LLM API**。  
2. **指对 K 线根目录**：例如  
   `/gpudata/crypto/data/parquet_data/futures/um/monthly/klines`  
   （支持**子目录**下放每个交易对的多个 parquet，见下文「数据目录」）。  
3. **跑一条命令**：`scripts/run_alphacrafter.py`，在终端看 **JSON 回测指标** + 可选 **LLM 交互日志**。

---

## 一、服务器上从头到尾怎么跑（按顺序复制）

以下假设：**Linux**、已能 `ssh` 登录、已安装 **conda** 与 **git**；项目在 **`~/projects/AlphaCrafter`**（与下文 `git clone` 路径一致），数据已在 **`/gpudata/crypto/data/parquet_data/futures/um/monthly/klines`**（只读即可）。

### 1）登录并进入工作目录

```bash
ssh <你的账号>@<服务器>
mkdir -p ~/projects && cd ~/projects
```

### 2）拉代码

```bash
git clone https://github.com/<你的用户名>/AlphaCrafter.git
cd AlphaCrafter
```

（没有 GitHub 时，用 `scp`/`rsync` 把本机整个项目目录拷到 `~/projects/AlphaCrafter` 亦可。）

### 3）Conda 环境与依赖

```bash
conda create -n alphacrafter python=3.11 -y
conda activate alphacrafter
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### 4）环境变量（必改 LLM）

```bash
cp .env.example .env
nano .env
```

至少保证（见第二节表格）：

- `OPENAI_API_KEY`（或你用的兼容接口 Key）  
- `ALPHACRAFTER_LLM_PROVIDER=openai`（DeepSeek 等）  
- 建议写死数据目录（与命令行二选一即可）：  
  `ALPHACRAFTER_CRYPTO_DATA_DIR=/gpudata/crypto/data/parquet_data/futures/um/monthly/klines`

### 5）确认数据目录可读

```bash
ls /gpudata/crypto/data/parquet_data/futures/um/monthly/klines | head
```

应能看到交易对子目录或 `.parquet` 文件。若权限不足，需找管理员给 `hm` 用户读权限或加入相应组。

### 6）自检（可选，不耗 API）

```bash
conda activate alphacrafter
cd ~/projects/AlphaCrafter
ALPHACRAFTER_LLM_PROVIDER=stub python -m unittest discover -s tests -p "test_*.py" -q
```

### 7）正式跑一轮（Table 1「回测」窗 + 本地 K 线）

```bash
conda activate alphacrafter
cd ~/projects/AlphaCrafter

python scripts/run_alphacrafter.py \
  --crypto-data-dir /gpudata/crypto/data/parquet_data/futures/um/monthly/klines \
  --split backtesting \
  --tickers 20 \
  --crypto-rank-by volume \
  --artifacts-dir runs/$(date +%Y%m%d_%H%M)
```

- 若已在 `.env` 里设置了 `ALPHACRAFTER_CRYPTO_DATA_DIR`，可省略 `--crypto-data-dir`，只保留 `--split` 等参数。  
- `--artifacts-dir` 可省略；需要 `equity_curves.png` 时请确保已 `pip install matplotlib`（见 `requirements.txt`）。  
- 长时间任务建议：`screen` / `tmux` 或 `nohup ... > run.log 2>&1 &`。

---

## 二、你需要改哪些「参数」（`.env` 与命令行）

### 必改（否则无法调真实大模型）

| 位置 | 变量 / 行为 | 说明 |
|------|-------------|------|
| `.env` | `OPENAI_API_KEY` | DeepSeek / 其它 OpenAI 兼容网关的密钥 |
| `.env` | `ALPHACRAFTER_LLM_PROVIDER` | 例如 `openai`；勿与已填的 `ANTHROPIC_API_KEY` 冲突时仍想用 DeepSeek，请显式设为 `openai` |

### 强烈建议改（决定用哪段数据、多少标的）

| 位置 | 变量 / CLI | 说明 |
|------|------------|------|
| `.env` 或 CLI | `ALPHACRAFTER_CRYPTO_DATA_DIR` / `--crypto-data-dir` | K 线根目录，**二选一即可** |
| CLI | `--split` | `training` / `validation` / `backtesting` / `live_trading`（论文 Table 1 日历切在 `date` 上过滤） |
| CLI | `--tickers` | 参与回测的标的数量上限；越大越慢 |
| `.env` 或 CLI | `ALPHACRAFTER_CRYPTO_RANK_BY` / `--crypto-rank-by` | `volume`：按近 N 日成交量和排序取前 `--tickers`；`none`：按扫描顺序取前 N |
| `.env` | `ALPHACRAFTER_CRYPTO_LOOKBACK_DAYS` | 排序用成交量窗口（默认 90 日历日） |
| `.env` | `ALPHACRAFTER_CRYPTO_SCAN_SUBDIRS` | 默认 `1`：递归扫子目录（适配 `.../klines/BTCUSDT/*.parquet`）；`0` 只扫根目录一层文件 |

### 常用调参（影响因子是否进库、回测探索范围）

| `.env` 变量 | 作用 |
|-------------|------|
| `ALPHACRAFTER_MINER_IC_ACCEPT` | Miner 接受因子 IC 下限（代码默认 **0.02**，可用 `.env` 收紧） |
| `ALPHACRAFTER_MINER_IC_RETAIN` | 维护库时低于则下架（默认 **0.008**） |
| `ALPHACRAFTER_MINER_MAX_ITERATIONS` | Miner 单次 `run` 内最多「生成→校验→记 H」次数（代码默认 **100**） |
| `ALPHACRAFTER_MINER_SEED_IC_RATIO` | 内置 seed 阈值 = `IC_ACCEPT × ratio`（代码默认 **0.38**） |
| `ALPHACRAFTER_TRADER_MAX_EXPLORATIONS` | Trader 网格尝试上限（代码默认 **12**） |
| `ALPHACRAFTER_MINER_IC_OOS_MIN` | 训练 IC 过线后，还需 **validation（2023）** 窗 IC ≥ 该值才写入 Z；默认 **0.0**（主要拒绝 OOS 执行失败 / NaN） |
| `ALPHACRAFTER_BARS_PER_YEAR` | 年化用 bar 数；不设且为 crypto 时默认 **365** |
| `ALPHACRAFTER_DISABLE_BUILTIN_SEED=1` | 禁止在 Z 为空时用内置因子 seed（更难跑通端到端，仅当你要严格只用 LLM 因子时） |

### 日志与落盘路径

| `.env` 变量 | 作用 |
|-------------|------|
| `ALPHACRAFTER_DATA_DIR` | 数据根；默认项目下 `data/`（SQLite、LLM 日志默认在此树下） |
| `ALPHACRAFTER_DB_PATH` | 共享内存 SQLite；不设则为 `<DATA_DIR>/shared_memory.db` |
| `ALPHACRAFTER_LLM_LOG` | `1` 开（默认）/ `0` 关 LLM JSONL |
| `ALPHACRAFTER_LLM_LOG_PATH` | 自定义 LLM 日志文件；不设则为 `<DATA_DIR>/logs/llm_interactions.jsonl` |
| `ALPHACRAFTER_LLM_LOG_PREVIEW_CHARS` | 每条 JSON 里 system/user/response 预览最大字符数 |

完整列表仍以 **`.env.example`** 为准。

**若 Miner 仍显示 `max_iterations=4` 或 `ic_accept=0.03`**：检查项目根目录 **`.env`** 是否仍写着旧值（`dotenv` 会覆盖代码默认）；删除或改为与 `.env.example` 一致后重跑。stderr 第一行会打印 `env ALPHACRAFTER_MINER_MAX_ITERATIONS=...` 便于核对。

**样本外 IC**：在 `--split validation|backtesting|live_trading` 且走 Table 1 下载面板时，Miner 只在 **training** 上算主 IC；**入库前**再在 **validation（2023）** 窗上算一次 IC，未过 `ALPHACRAFTER_MINER_IC_OOS_MIN` 则记为 `ineffective_oos`、**不进入 Z**（见 JSON `library_discipline.miner_oos_*`）。

---

## 三、跑完后「结论」怎么看

### 1）终端里的回测结果（主结论）

`python scripts/run_alphacrafter.py ...` 会打印：

- **`training_panel_diagnostics`**（`training` 或 eval 段）：训练窗（或 Table 1 训练窗上的面板）**日历覆盖**与 `warnings` 提示（如历史太短、起始日晚于 2016 等），便于发现「IC 挖不动」是否与数据跨度有关。  

- 一大段 **`ok`: true/false** 的 **JSON**。重点字段：  
  - **`benchmark.metrics`**：当前评估窗上的 **等权多头基准**（`sharpe_ann`、`ann_return_pct`、`max_drawdown_pct` 等），**不依赖** Trader 是否成功。  
  - **`trader.best_metrics`**（若 `trader` 非 `null`）：策略搜索最优的 `sharpe_ann`、`ann_return_pct`、`max_drawdown_pct`、`cum_return`、`n`  
  - **`trader.best_spec`**：本次最优的 clip/gross/net_bias  
  - **`trader.equity_curve`**（若有）：日频权益曲线，便于落盘或再画图  
  - **`artifacts`**（若命令行传了 `--artifacts-dir`）：`summary.json`、基准 CSV、可选 `equity_curves.png` 等路径  
  - **`regime`**、**`ensemble_id`**、**`miner`** / **`miner_seed`**：辅助判断因子是否主要来自 LLM 或 seed  
  - **`library_discipline`**：是否 `paper_eval`（eval 段不写 Z）等  

- JSON 之后若未加 `--json-only`，stderr 上还有 Miner 每步进度（`ALPHACRAFTER_MINER_VERBOSE=0` 可关）；stdout 上会多几行摘要：先 **等权基准** 的 **`AR(%)` / `SR` / `MDD(%)`**，若有 Trader 再打印策略回测的同样三项。

可把整段输出重定向留档：

```bash
python scripts/run_alphacrafter.py ... 2>&1 | tee run_$(date +%F_%H%M).log
```

### 2）大模型交互过程（Miner / Screener）

默认写入 **JSON Lines**（一行一次 `complete_text` 调用）：

```bash
# 默认路径（未改 ALPHACRAFTER_DATA_DIR 时）
tail -n 20 ~/projects/AlphaCrafter/data/logs/llm_interactions.jsonl
```

每行 JSON 含：`ts_iso`、`agent`（`miner` / `screener`）、`provider`、`model`、`system_preview`、`user_preview`、`response_preview` 等。  
若装了 `jq`，可按智能体过滤：

```bash
grep '"agent": "miner"' data/logs/llm_interactions.jsonl | tail -n 3
```

### 3）SQLite 里的结构化记录（H）

默认库文件：**`<DATA_DIR>/shared_memory.db`**（与上表一致）。可用 [DB Browser for SQLite](https://sqlitebrowser.org/) 或：

```bash
sqlite3 data/shared_memory.db ".tables"
```

表内含因子事件、策略候选、执行记录等（随 Miner/Screener/Trader 写入）。

### 4）若 `ok`: false 或 `empty_panel`

- **日期与数据是否重叠**：Table 1 的 `training` 为 2016–2022，若 K 线只有近几年，训练窗会空 → 可先试 **`--split training`** 且保证数据覆盖该区间，或暂时用**无 `--split`** 的滚动窗口（仅用最近 `--days` 日历跨度，见代码 `run_pipeline` 的 rolling 分支）。  
- **目录是否扫到文件**：确认 `ALPHACRAFTER_CRYPTO_SCAN_SUBDIRS=1`，且路径下确有 `.parquet`/`.csv`。  
- **列名**：需含时间列 + OHLC（volume 可缺省为 0）。  

### 5）诊断图：Miner 迭代 IC、Trader 试探夏普、因子样本内外 IC、组合夏普

脚本 **`scripts/plot_diagnostics.py`** 从 **`shared_memory.db`** 读最近一次管线写入的记录，生成 PNG（需 **matplotlib**）。

| 输出文件 | 含义 |
|----------|------|
| `miner_ic_by_attempt.png` | 最近 `--tail` 条因子尝试：**横轴 = 时间顺序上的第几次尝试**，纵轴 = IC；橙色虚线 = **历史最佳 IC 随尝试累积上升**（体现「随迭代变好」的上界轨迹）。 |
| `miner_ic_vs_ir_scatter.png` | 同上批次：**IC vs IR 散点**，颜色 = `effective` / `ineffective` / `deprecated`。 |
| `trader_sharpe_by_trial.png` | 当前最新 **ensemble** 下各次 **Trader 探索**：横轴 = 试探序号，纵轴 = 该次在**评估窗**上的年化夏普；橙色 = **best-so-far**（策略参数搜索过程）。 |
| `factor_ic_train_vs_<oos>.png` | 需 `--crypto-data-dir`：因子库 **Z** 中每个因子在 **training** 与 **validation/backtesting** 窗上**重新算 IC** 的散点（样本内 vs 样本外 IC）。 |
| `portfolio_sharpe_train_vs_<oos>.png` | 需 `--crypto-data-dir` 且库内有 ensemble + 候选：用 DB 里**得分最高的 spec**，在同一 ensemble 信号下算 **训练窗 vs OOS 窗** 各一个夏普点（单点对照图）。 |

**注意**：`--tail` 只取 `factor_records` **按 id 倒序的最后 N 条**；若多次跑管线共用同一库，图里会混入历史尝试——做对比实验可为每次运行设独立 `ALPHACRAFTER_DB_PATH`。

在服务器上（先跑完至少一次 `run_alphacrafter.py` 再画图），**逐行**执行：

```bash
cd ~/projects/AlphaCrafter
```

```bash
conda activate alphacrafter
```

```bash
python scripts/plot_diagnostics.py \
  --out-dir runs/diagnostics_plots \
  --tail 200
```

```bash
python scripts/plot_diagnostics.py \
  --db data/shared_memory.db \
  --out-dir runs/diagnostics_plots_full \
  --tail 500 \
  --crypto-data-dir /gpudata/crypto/data/parquet_data/futures/um/monthly/klines \
  --tickers 20 \
  --crypto-rank-by volume \
  --oos-split backtesting
```

```bash
ls -la runs/diagnostics_plots_full
```

```bash
# 若 SSH 客户端支持 SFTP，在本机拉取图片；或把目录打成压缩包
cd ~/projects/AlphaCrafter && tar czvf runs/diag_png.tgz runs/diagnostics_plots_full/*.png
```

（OOS 用 **validation（2023）** 时把上一段里的 `--oos-split backtesting` 改成 `--oos-split validation` 即可。）

### 6）论文式两阶段 + 诊断图（推荐）

在同一 `shared_memory.db` 上先 **训练窗填满 Z**，再 **回测窗评估**（避免只在 `backtesting` 单跑、eval 纪律下 seed 不写库导致 Z 长期为空）：

```bash
cd ~/projects/AlphaCrafter && conda activate alphacrafter
bash scripts/run_crypto_paper_workflow.sh /gpudata/crypto/data/parquet_data/futures/um/monthly/klines 20
```

第三个参数可指定产物根目录，例如：`bash scripts/run_crypto_paper_workflow.sh "$CRYPTO_DIR" 20 runs/my_exp_1`

---

## 四、数据目录约定（与你的 `.../monthly/klines`）

- **递归扫描**（默认开启）：会收集根目录及所有子目录下的 `*.parquet` / `*.csv` / `*.pq`。  
- **标的推断**：若文件在子目录中，例如 `.../klines/BTCUSDT/xxx.parquet`，**交易对名取子目录名 `BTCUSDT`**。若文件全在根目录且文件名为 `BTCUSDT-1d-2024-01.parquet` 这类，会取 **`BTCUSDT-`** 前缀作为标的。  
- **同一标的多文件**（如多个月度 parquet）：在相同 `date` 上会 **合并 OHLCV**（volume 相加，high/max、low/min 等），再进入后续截面逻辑。

若你的实际路径比上述更深一层，只要把 **`--crypto-data-dir` 指到仍包含各 `SYMBOL/` 子目录的那一层**即可。

---

## 五、多智能体流程与论文对齐（简述）

| 智能体 | 论文角色 | LLM？ | 主要职责 |
|--------|----------|--------|----------|
| **Miner** | 因子发现 | 是 | 生成因子代码 → IC/IR → 更新 **Z** / **H** |
| **Screener** | 体制 + 组合 | 是 | **M** → **R̂** → 从 **Z** 得 **E** → 写 **H** |
| **Trader** | 策略与回测 | 否 | 网格 + 向量化回测 → **π** → 写 **H** |

**Table 1**：`validation` / `backtesting` / `live_trading` 时，**仅在 `training` 日历窗上更新 Z**；当前评估窗只做 Screener+Trader，避免用未来数据扩库（`miner_seed.reason`：`eval_phase_no_Z_writes`）。

- **与论文「训练挖因子」是否一致**：一致。上述 eval 阶段下，**Miner 只在 `training`（2016–2022）面板上循环**（生成因子 → IC/IR → 写入 **H**，达标写入 **Z**）；**不会在 validation/backtesting 窗上改 Z**。  
- **「测试集 / 回测窗」**：Table 1 里 **`validation`（2023）** 与 **`backtesting`（2024–2025）** 是两个不同评估窗；`--split backtesting` 表示在 **回测窗** 上跑 Screener+Trader（因子仍只来自训练窗 Miner）。不是把「测试集」混进训练去挖因子。  
- **「每次迭代都能优化」**需区分两层：  
  1. **Miner**：单次进程里 `ALPHACRAFTER_MINER_MAX_ITERATIONS` 次迭代，**每一步都在训练窗上**尝试新因子并记录；**不是**在回测窗上迭代改库。  
  2. **Trader**：在**当前 split 对应的面板**（例如 backtesting）上做多次网格试探，相当于在**该窗内**搜索策略参数 **π**（多步「优化」夏普等）；**不**更新因子库 Z。  
  3. **非 walk-forward**：单次 `run_alphacrafter.py` 不会在日历上逐日滚动；若要「日复一日」闭环，需自行定时多次运行或另写外层调度。

编排入口：`alphacrafter/orchestration/loop.py`。

---

## 六、Windows 本机快速试跑（可选）

```powershell
cd e:\project\AlphaCrafter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
python scripts\run_alphacrafter.py --crypto-data-dir D:\demo_klines --split training --tickers 5 --crypto-rank-by none
```

---

## 七、推送到 GitHub（逐行命令）

在**本机**或服务器上、已在项目根目录初始化过 git 的前提下（`git status` 无报错）：

```bash
cd ~/projects/AlphaCrafter
```

```bash
git status
```

```bash
git add -A
```

```bash
git diff --cached --stat
```

```bash
git commit -m "feat: miner/trader defaults, training panel hints, crypto workflow script"
```

若远程尚未配置（只需做一次）：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
```

若已存在 `origin` 且 URL 正确，可跳过上一行。推送主分支：

```bash
git branch -M main
```

```bash
git push -u origin main
```

若仓库默认分支是 `master`，把上面两行里的 `main` 改成 `master`。  
首次推送若开启 2FA，需用 **Personal Access Token** 代替密码，或在服务器上配置 **SSH**：`git remote set-url origin git@github.com:<用户>/<仓库>.git` 后再 `git push`.

---

## 八、附录

- **单元测试**：`python -m unittest discover -s tests -p "test_*.py" -v`  
- **诊断图脚本自检**：`python -m unittest tests.test_plot_diagnostics -v`  
- **panel 提示**：`python -m unittest tests.test_panel_hints -v`  
- **旧美股 Yahoo + 维基**：仍可用 `scripts/scrape_sp500.py` + `run_alphacrafter.py --universe ...`，非当前文档主线。

---

## 免责声明

本仓库用于研究与教育目的。任何回测结果不构成投资建议；实盘交易需自行承担风险并遵守当地法规与交易所规则。
