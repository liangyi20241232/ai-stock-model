# A股 AI算力与半导体股票池预测系统

一个面向A股 AI 算力与半导体主线的本地量化研究项目。项目使用 AKShare 免费日线行情、本地 CSV / Parquet 文件、LightGBM 分类模型和可配置仓位规则，预测股票未来 20 个交易日跑赢 AI 股票池平均收益的概率，并生成建议仓位、简化回测和 Markdown 报告。

> 本项目仅用于个人学习、量化研究和代码实验，不构成任何投资建议。模型结果不能直接用于实盘交易，回测不代表未来收益。

## 项目特点

- 聚焦A 股 AI 算力与半导体主线。
- 股票池支持 `core_candidate` 和 `satellite_candidate` 分层。
- 第一版只使用日线行情，不使用 Docker、不使用 GPU、不使用深度学习。
- 数据优先通过 AKShare 获取，失败时会尝试多个免费行情源。
- 所有行情和结果都保存在本地，不依赖复杂数据库。
- 模型优先使用 LightGBM，失败时自动 fallback 到随机森林或逻辑回归。
- 输出的是 `p_outperform`，即未来 20 个交易日跑赢 AI 股票池平均收益的概率，不是上涨概率。
- 仓位规则可通过 `config/settings.yaml` 修改。
- 自动生成预测、仓位、回测和模型解释报告。
- 包含 2026 年特征重要性和最新一期预测解释。

## 研究目标

完整流程如下：

```text
AI 股票池名单
→ 拉取日线行情
→ 生成特征
→ 生成标签
→ 训练模型
→ 输出 p_outperform
→ 生成建议仓位
→ 简化回测
→ 生成 Markdown 报告
```

当前预测目标：

```text
未来 20 个交易日，某只股票是否跑赢 AI 股票池等权平均收益。
```

标签定义：

```text
future_ret_20 = T+20 收盘价 / T+1 开盘价 - 1
future_ai_ret_20 = 同期 AI 股票池平均收益
label = 1 if future_ret_20 > future_ai_ret_20 else 0
```

## 只关注 AI 算力与半导体

只关注和 AI 主线直接相关的 A 股方向，例如：

- AI 算力
- 服务器
- PCB
- 光模块 / CPO
- 液冷
- 电源
- 数据中心设备
- AI 芯片
- GPU 产业链
- 存储
- 先进封装 / 封测
- 半导体设备
- 半导体材料
- EDA
- 国产算力生态

银行、白酒、传统消费、地产、煤炭、石油、公用事业、新能源车、机器人、创新药、军工等不进入可投资池，可以作为风格监控变量。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── main.py
├── install_deps.sh
├── run_pipeline.sh
├── fix_lightgbm_libomp.sh
├── config/
│   ├── ai_pool.csv
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_get_data.ipynb
│   ├── 02_build_features.ipynb
│   ├── 03_train_model.ipynb
│   └── 04_backtest.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── labels.py
│   ├── model.py
│   ├── explain.py
│   ├── portfolio.py
│   ├── backtest.py
│   └── report.py
└── output/
```

说明：

- `config/`：人工配置区，包括股票池和策略参数。
- `data/raw/`：原始日线行情。
- `data/processed/`：加工后的特征表和训练集。
- `src/`：项目源码。
- `output/`：预测、仓位、回测和报告结果。
- `notebooks/`：给新手学习和拆解流程用的占位 notebook。

`data/raw/`、`data/processed/`、`output/` 和 `.venv/` 默认不会上传 GitHub。

## 快速开始

### 1. 安装依赖

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

也可以直接运行脚本：

```bash
bash install_deps.sh
```

如果 Mac 上 LightGBM 报 `libomp.dylib` 相关错误，可以运行：

```bash
bash fix_lightgbm_libomp.sh
```

### 2. 一键运行完整流程

```bash
bash run_pipeline.sh
```

或者：

```bash
python main.py run-all
```

### 3. 分步骤运行

```bash
python main.py fetch-data
python main.py build-features
python main.py train
python main.py portfolio
python main.py backtest
python main.py report
```

## 股票池配置

股票池文件：

```text
config/ai_pool.csv
```

字段说明：

- `stock_code`：6 位股票代码，例如 `000977`，前导 0 不能丢。
- `stock_name`：股票名称。
- `theme_1`：主主题。
- `theme_2`：副主题。
- `layer`：只能填 `core_candidate` 或 `satellite_candidate`。

层级含义：

- `core_candidate`：AI 算力或半导体主线更明确、产业位置更核心、资金关注度可能更高的候选股。
- `satellite_candidate`：仍然和 AI 主线直接相关，但确定性、资金集中度或产业位置相对低一些的候选股。

当前股票池只是示例，不代表投资建议。股票池需要人工维护，建议每 1-3 个月检查一次。

## 策略参数

参数文件：

```text
config/settings.yaml
```

主要参数：

- `entry_threshold`：进入仓位的最低概率阈值。默认 `0.55`。
- `full_confidence`：接近满额仓位的概率阈值。默认 `0.70`。
- `probability_gamma`：控制仓位曲线的非线性程度。默认 `2.0`。
- `max_total_weight`：组合总仓位上限。默认 `0.70`。
- `core_budget_ratio`：核心层预算比例。默认 `0.70`。
- `satellite_budget_ratio`：卫星层预算比例。默认 `0.30`。
- `core_single_cap`：核心候选单只最高仓位。默认 `0.10`。
- `satellite_single_cap`：卫星候选单只最高仓位。默认 `0.03`。

之前提到过的 52%、12%、4%、80% 不是第一版正式规则。第一版统一使用 `settings.yaml` 中的 55%、70%、10%、3%、70%，方便先跑通闭环，后续再基于回测和观察逐步调参。

## 当前特征

模型当前使用的日线特征包括：

- `ret_5`
- `ret_20`
- `ret_60`
- `ret_120`
- `volume_chg_20`
- `amount_chg_20`
- `ma_20_gap`
- `ma_60_gap`
- `vol_20`
- `vol_60`
- `drawdown_60`
- `high_60_breakout`
- `high_120_breakout`
- `relative_strength_20`
- `relative_strength_60`

这些特征不是世界公认的唯一标准，而是第一版的基础量价特征。后续可以继续加入新的交易假设，例如量价比、换手率、放量突破、行业相对强弱、市场风格变量等。

## 输出文件

运行完成后会生成：

- `data/raw/fetch_log.csv`：行情拉取日志。
- `data/processed/features.parquet`：特征数据。
- `data/processed/dataset.parquet`：带标签的训练数据。
- `output/predictions.csv`：历史测试期和最新一期预测结果。
- `output/target_weights.csv`：每个信号日的建议目标仓位。
- `output/backtest_result.csv`：回测指标。
- `output/equity_curve.csv`：回测净值曲线。
- `output/report.md`：主报告。
- `output/feature_importance_2026.csv`：2026 年特征重要性。
- `output/latest_explain_2026.csv`：最新一期预测解释。
- `output/model_explain_report.md`：模型解释报告。

日常最建议查看：

```text
output/report.md
output/model_explain_report.md
```

## 补充文档

- [使用指南](docs/usage.md)
- [模型说明](docs/model_notes.md)

## 如何理解模型解释

`output/model_explain_report.md` 中会展示 2026 年特征重要性和方向相关性。

- `重要性` 回答的是：打乱这个特征后，模型表现会不会明显变差。它用来判断这个特征有没有被模型有效利用。
- `方向相关性` 回答的是：这个特征偏高时，在 2026 年样本里更常对应跑赢还是跑输。它用来粗略判断“偏高更有利”还是“偏低更有利”。
- `方向相关性 = 0` 或接近 0，不等于这个特征没用，只表示它没有明显的单向线性关系。
- 判断“打乱这个值有没有关系”，主要看 `重要性`。
- 判断“这个值越高越好还是越低越好”，才看 `方向相关性`。

模型解释不是因果结论，只能作为研究线索。

## 第一版限制

- 股票池是人工维护的，可能有遗漏或错误。
- AKShare 数据可能存在缺失、延迟或接口变化。
- 第一版只使用日线行情，没有使用财务数据、公告、研报、资金流、融资融券、龙虎榜、机构持仓等信息。
- 模型输出概率未经严格概率校准，不一定是真实概率。
- 回测不处理涨停买不进、跌停卖不出。
- 回测使用 T 日收盘信号、T+1 开盘交易的简化逻辑。
- 历史预测来自一次时间切分后的测试期模型，不是逐月滚动重新训练。
- 交易成本只做简单单边成本估算。
- 回测不能代表未来收益，不能直接用于实盘交易。

## 常见问题

### 这个项目预测的是上涨概率吗？

不是。项目预测的是：

```text
未来 20 个交易日跑赢 AI 股票池平均收益的概率。
```

字段名是：

```text
p_outperform
```

### 股票池需要每天修改吗？

不需要。股票池是候选范围，建议每 1-3 个月检查一次。每天或每周更新的是行情数据和模型输出。

### 我每天应该做什么？

如果只是观察，可以一周运行 1-2 次：

```bash
bash run_pipeline.sh
```

如果用于月度研究，建议每月最后一个交易日收盘后重点运行一次，并查看报告。

### 如果运行报错，先检查什么？

1. `config/ai_pool.csv`：股票代码是否是 6 位，`layer` 是否正确。
2. `config/settings.yaml`：缩进是否正确，参数是否非法。
3. `data/raw/fetch_log.csv`：哪些股票拉取失败。
4. `output/predictions.csv`：是否包含 `p_outperform`。
5. LightGBM 如果报 `libomp.dylib`，运行 `bash fix_lightgbm_libomp.sh`。

## Roadmap

- 增加数据质量检查报告。
- 增加滚动训练回测。
- 增加更多日线特征。
- 增加 SHAP 或更细粒度的模型解释。
- 增加股票池维护辅助工具。
- 增加自定义基准。
- 增加市场风格监控变量。
- 优化概率校准。

## 免责声明

本项目仅用于个人学习、量化研究和代码实验，不构成任何投资建议、投资咨询或交易推荐。项目中的股票池、模型输出、建议仓位和回测结果都不能直接用于实盘交易。

金融市场具有高度不确定性，历史回测不代表未来收益。使用者应自行承担所有风险。
