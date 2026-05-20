# 使用指南

本文档记录本项目的日常使用方式。项目主入口是 `main.py`，推荐新手优先使用 `run_pipeline.sh`。

## 第一次运行

1. 进入项目目录：

```bash
cd /Users/daniel7712/Documents/Codex/2026-05-17/python-a-ai-1-2-macbook
```

2. 安装依赖：

```bash
bash install_deps.sh
```

3. 如果 LightGBM 报 `libomp.dylib`：

```bash
bash fix_lightgbm_libomp.sh
```

4. 运行完整流程：

```bash
bash run_pipeline.sh
```

## 日常更新

每次想更新数据和报告时，运行：

```bash
bash run_pipeline.sh
```

这个命令会依次执行：

```text
拉取行情数据
→ 生成特征和标签
→ 训练模型并预测
→ 生成建议仓位
→ 运行简化回测
→ 生成 Markdown 报告
```

## 推荐查看的输出

- `output/report.md`：主报告，包含最新预测、建议持仓、回测摘要。
- `output/model_explain_report.md`：模型解释报告，包含 2026 年特征重要性和最新一期解释。
- `output/predictions.csv`：每只股票的 `p_outperform`。
- `output/target_weights.csv`：建议目标仓位。

## 股票池维护

股票池文件是：

```text
config/ai_pool.csv
```

建议每 1-3 个月检查一次，不需要每天修改。只加入和 AI 算力、半导体主线直接相关的 A 股。

`layer` 只能使用：

- `core_candidate`
- `satellite_candidate`

## 参数调整

参数文件是：

```text
config/settings.yaml
```

常用参数：

- `entry_threshold`：进入仓位的最低概率阈值。
- `full_confidence`：接近满额仓位的概率阈值。
- `probability_gamma`：仓位曲线的非线性程度。
- `max_total_weight`：组合总仓位上限。
- `core_single_cap`：核心层单只上限。
- `satellite_single_cap`：卫星层单只上限。

建议一次只改少数参数，改完后重新运行完整流程，再观察输出变化。

## 常见问题

### 预测结果是不是上涨概率？

不是。`p_outperform` 表示未来 20 个交易日跑赢 AI 股票池平均收益的概率。

### 为什么报告里没有建议持仓？

可能是所有股票的 `p_outperform` 都没有超过 `entry_threshold`。这不一定是错误，说明模型当前不愿意给仓位。

### AKShare 拉取失败怎么办？

先查看：

```text
data/raw/fetch_log.csv
```

如果是网络或接口临时问题，稍后重跑即可。

## 风险提示

本项目仅用于学习和研究，不构成任何投资建议。模型结果未经严格概率校准，回测不能代表未来收益，不能直接用于实盘交易。
