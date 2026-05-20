from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_settings() -> dict[str, Any]:
    """读取报告需要展示的关键配置。"""
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt_pct(value: float | int | None) -> str:
    """把小数格式化为百分比，空值显示为 N/A。"""
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def _latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """取最新日期的数据。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    return df[df["date"] == latest_date].copy()


def generate_report() -> Path:
    """生成 output/report.md Markdown 报告。"""
    settings = load_settings()
    predictions_path = OUTPUT_DIR / "predictions.csv"
    weights_path = OUTPUT_DIR / "target_weights.csv"
    backtest_path = OUTPUT_DIR / "backtest_result.csv"
    explain_path = OUTPUT_DIR / "feature_importance_2026.csv"

    if not predictions_path.exists():
        raise FileNotFoundError("找不到 output/predictions.csv，请先运行：python main.py train")
    if not weights_path.exists():
        raise FileNotFoundError("找不到 output/target_weights.csv，请先运行：python main.py portfolio")

    predictions = pd.read_csv(predictions_path, dtype={"stock_code": str})
    weights = pd.read_csv(weights_path, dtype={"stock_code": str})
    latest_pred = _latest_rows(predictions).sort_values("p_outperform", ascending=False).head(20)
    latest_weights = _latest_rows(weights)
    latest_weights = latest_weights[latest_weights["target_weight"] > 0].sort_values("target_weight", ascending=False)

    if backtest_path.exists():
        metrics = pd.read_csv(backtest_path)
        metric_map = dict(zip(metrics["metric"], metrics["value"]))
    else:
        metric_map = {}

    cfg = settings["portfolio"]
    model_cfg = settings["model"]
    data_cfg = settings["data"]

    lines: list[str] = []
    lines.append("# AI 算力与半导体股票池预测报告")
    lines.append("")
    lines.append("本报告由最小可运行版本自动生成，仅用于学习和研究，不构成投资建议。")
    lines.append("")

    lines.append("## 最新一期预测前 20 名")
    lines.append("")
    lines.append("| 排名 | 日期 | 股票代码 | 股票名称 | 层级 | 跑赢概率 p_outperform |")
    lines.append("|---:|---|---|---|---|---:|")
    for rank, row in enumerate(latest_pred.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.date} | {row.stock_code} | {row.stock_name} | {row.layer} | {_fmt_pct(row.p_outperform)} |"
        )
    lines.append("")

    lines.append("## 最新一期建议持仓")
    lines.append("")
    if latest_weights.empty:
        lines.append("当前没有股票达到入场阈值，因此建议仓位为空。")
    else:
        lines.append("| 日期 | 股票代码 | 股票名称 | 层级 | p_outperform | p_score | 目标仓位 |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in latest_weights.itertuples(index=False):
            lines.append(
                f"| {row.date} | {row.stock_code} | {row.stock_name} | {row.layer} | "
                f"{_fmt_pct(row.p_outperform)} | {_fmt_pct(row.p_score)} | {_fmt_pct(row.target_weight)} |"
            )
    lines.append("")

    lines.append("## 回测结果摘要")
    lines.append("")
    if metric_map:
        lines.append(f"- 年化收益：{_fmt_pct(metric_map.get('annualized_return'))}")
        lines.append(f"- 最大回撤：{_fmt_pct(metric_map.get('max_drawdown'))}")
        lines.append(f"- 胜率：{_fmt_pct(metric_map.get('win_rate'))}")
        lines.append(f"- 月度胜率：{_fmt_pct(metric_map.get('monthly_win_rate'))}")
        lines.append(f"- 夏普比率：{metric_map.get('sharpe_ratio', float('nan')):.3f}")
        lines.append(f"- 换手率：{_fmt_pct(metric_map.get('turnover'))}")
        lines.append(f"- 总收益：{_fmt_pct(metric_map.get('total_return'))}")
        lines.append(f"- 相对 AI 股票池等权基准超额收益：{_fmt_pct(metric_map.get('excess_return_vs_ai_equal_weight'))}")
    else:
        lines.append("尚未生成回测结果。请先运行：`python main.py backtest`")
    lines.append("")

    lines.append("## 模型使用的特征列表")
    lines.append("")
    for col in FEATURE_COLUMNS:
        lines.append(f"- `{col}`")
    lines.append("")

    lines.append("## 2026 年模型解释")
    lines.append("")
    if explain_path.exists():
        importance = pd.read_csv(explain_path)
        if importance.empty:
            lines.append("2026 年可解释样本太少，暂时无法计算稳定的特征重要性。")
        else:
            lines.append("完整解释见：`output/model_explain_report.md`")
            lines.append("")
            lines.append("| 排名 | 特征 | 重要性 | 方向相关性 |")
            lines.append("|---:|---|---:|---:|")
            positive_importance = importance[importance["importance"] > 0].head(8)
            for rank, row in enumerate(positive_importance.itertuples(index=False), start=1):
                lines.append(
                    f"| {rank} | `{row.feature}` | {row.importance:.6f} | {row.direction_corr_2026:.4f} |"
                )
            if positive_importance.empty:
                lines.append("| - | 暂无正向稳定重要特征 | 0.000000 | 0.0000 |")
    else:
        lines.append("尚未生成 2026 年模型解释。请先运行：`python main.py train`")
    lines.append("")

    lines.append("## 当前关键参数")
    lines.append("")
    lines.append(f"- 预测周期：未来 {model_cfg.get('prediction_horizon', 20)} 个交易日")
    lines.append(f"- 数据起始日期：{data_cfg.get('start_date')}")
    lines.append(f"- 入场阈值 entry_threshold：{cfg.get('entry_threshold')}")
    lines.append(f"- 满信心阈值 full_confidence：{cfg.get('full_confidence')}")
    lines.append(f"- 仓位曲线参数 probability_gamma：{cfg.get('probability_gamma')}")
    lines.append(f"- 组合总仓位上限 max_total_weight：{cfg.get('max_total_weight')}")
    lines.append(f"- 核心层预算比例 core_budget_ratio：{cfg.get('core_budget_ratio')}")
    lines.append(f"- 卫星层预算比例 satellite_budget_ratio：{cfg.get('satellite_budget_ratio')}")
    lines.append(f"- 核心层单只上限 core_single_cap：{cfg.get('core_single_cap')}")
    lines.append(f"- 卫星层单只上限 satellite_single_cap：{cfg.get('satellite_single_cap')}")
    lines.append("")

    lines.append("## 风险提示")
    lines.append("")
    lines.append("- 股票池是人工维护的，可能有遗漏或错误。")
    lines.append("- AKShare 数据可能存在缺失、延迟或接口变化。")
    lines.append("- 第一版模型没有使用财务数据、公告、研报、资金流等信息。")
    lines.append("- 模型输出概率未经严格概率校准，不一定是真实概率。")
    lines.append("- 回测不能代表未来收益。")
    lines.append("- 本系统不能直接用于实盘交易。")
    lines.append("")

    lines.append("## 第一版模型限制")
    lines.append("")
    lines.append("- 回测不处理涨停买不进、跌停卖不出。")
    lines.append("- 历史预测来自一次时间切分后的测试期模型，不是逐月滚动训练。")
    lines.append("- 股票池行业归属需要你人工检查和维护。")
    lines.append("- 第一版只使用日线行情，没有接入基本面、公告、资金流或情绪数据。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已保存：{output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()
