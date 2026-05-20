from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"


def _safe_corr(feature: pd.Series, label: pd.Series) -> float:
    """计算特征和标签的相关方向，无法计算时返回 0。"""
    feature = pd.to_numeric(feature, errors="coerce")
    label = pd.to_numeric(label, errors="coerce")
    valid = feature.notna() & label.notna()
    if valid.sum() < 5 or feature[valid].nunique() < 2 or label[valid].nunique() < 2:
        return 0.0
    corr = feature[valid].corr(label[valid])
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _score_type_for_year(year_df: pd.DataFrame) -> str:
    """2026 年样本如果标签有两类，就用 AUC；否则用 accuracy。"""
    if year_df["label"].nunique(dropna=True) >= 2:
        return "roc_auc"
    return "accuracy"


def _calculate_base_score(model: Pipeline, x: pd.DataFrame, y: pd.Series, score_type: str) -> float:
    """计算 2026 年解释样本上的基础分数。"""
    if len(x) == 0:
        return np.nan
    if score_type == "roc_auc":
        p = model.predict_proba(x)[:, 1]
        return float(roc_auc_score(y, p))
    pred = model.predict(x)
    return float(accuracy_score(y, pred))


def build_2026_feature_importance(
    model: Pipeline,
    data: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    random_state: int,
) -> pd.DataFrame:
    """只用 2026 年已有标签样本，计算特征重要性。

    这里使用 permutation importance：把某个特征打乱，看模型分数下降多少。
    分数下降越多，说明这个特征在 2026 年样本里越重要。
    """
    year_df = data.copy()
    year_df["date"] = pd.to_datetime(year_df["date"])
    year_df = year_df[(year_df["date"].dt.year == 2026) & year_df["label"].notna()].copy()
    year_df = year_df.dropna(subset=feature_cols, how="all")

    if len(year_df) < 20:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "raw_importance",
                "importance_std",
                "direction_corr_2026",
                "score_type",
                "base_score",
                "sample_count",
                "model_name",
            ]
        )

    x_year = year_df[feature_cols]
    y_year = year_df["label"].astype(int)
    score_type = _score_type_for_year(year_df)
    base_score = _calculate_base_score(model, x_year, y_year, score_type)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        result = permutation_importance(
            model,
            x_year,
            y_year,
            n_repeats=10,
            random_state=random_state,
            scoring=score_type,
            n_jobs=1,
        )

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "raw_importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    # permutation importance 偶尔会出现负数，通常代表该特征在这段样本里不稳定或没有帮助。
    # 报告排序和单股解释使用非负 importance，原始值保留在 raw_importance 里方便追踪。
    importance["importance"] = importance["raw_importance"].clip(lower=0)
    importance["direction_corr_2026"] = [
        _safe_corr(year_df[col], year_df["label"]) for col in feature_cols
    ]
    importance["score_type"] = score_type
    importance["base_score"] = base_score
    importance["sample_count"] = len(year_df)
    importance["model_name"] = model_name
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


def _feature_reason_text(feature: str, value: float, z_score: float, direction: float) -> str:
    """把单个特征的简化解释翻译成人话。"""
    if direction > 0:
        relation = "偏高时在 2026 年样本里更常对应跑赢"
    elif direction < 0:
        relation = "偏低时在 2026 年样本里更常对应跑赢"
    else:
        relation = "方向暂不稳定"

    side = "高于" if z_score >= 0 else "低于"
    return f"`{feature}` 当前值 {value:.4f}，{side} 2026 年中位水平，{relation}"


def build_latest_explain_table(
    latest: pd.DataFrame,
    data: pd.DataFrame,
    importance: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """为最新一期预测前 N 名生成简化原因。

    这不是 SHAP，也不是因果解释；它只是把“2026 年重要性、方向、当前特征位置”
    组合成容易阅读的提示。
    """
    if importance.empty:
        return pd.DataFrame(columns=["stock_code", "stock_name", "p_outperform", "reason_1", "reason_2", "reason_3"])

    data_2026 = data.copy()
    data_2026["date"] = pd.to_datetime(data_2026["date"])
    data_2026 = data_2026[data_2026["date"].dt.year == 2026]
    medians = data_2026[feature_cols].median(numeric_only=True)
    stds = data_2026[feature_cols].std(numeric_only=True).replace(0, np.nan)

    imp_map = importance.set_index("feature")["importance"].clip(lower=0)
    if imp_map.sum() <= 0:
        imp_map = pd.Series(1.0, index=feature_cols)
    imp_map = imp_map / imp_map.sum()
    direction_map = importance.set_index("feature")["direction_corr_2026"]

    rows = []
    latest = latest.sort_values("p_outperform", ascending=False).head(top_n)
    for row in latest.itertuples(index=False):
        row_dict = row._asdict()
        scored_reasons: list[tuple[float, str]] = []
        for feature in feature_cols:
            value = row_dict.get(feature)
            if pd.isna(value) or feature not in medians.index or pd.isna(stds.get(feature)):
                continue
            z_score = float((value - medians[feature]) / stds[feature])
            direction = float(direction_map.get(feature, 0.0))
            reason_score = abs(z_score) * float(imp_map.get(feature, 0.0))
            reason = _feature_reason_text(feature, float(value), z_score, direction)
            scored_reasons.append((reason_score, reason))

        top_reasons = [text for _, text in sorted(scored_reasons, reverse=True)[:3]]
        while len(top_reasons) < 3:
            top_reasons.append("暂无足够稳定的解释")

        rows.append(
            {
                "stock_code": str(row_dict["stock_code"]).zfill(6),
                "stock_name": row_dict["stock_name"],
                "p_outperform": row_dict["p_outperform"],
                "reason_1": top_reasons[0],
                "reason_2": top_reasons[1],
                "reason_3": top_reasons[2],
            }
        )
    return pd.DataFrame(rows)


def write_2026_explain_report(
    importance: pd.DataFrame,
    latest_explain: pd.DataFrame,
    output_path: Path,
) -> Path:
    """把 2026 年特征重要性和最新一期解释写成 Markdown。"""
    lines: list[str] = []
    lines.append("# 2026 年模型解释报告")
    lines.append("")
    lines.append("本报告只解释 2026 年已有标签样本，不再做不同年份对比。")
    lines.append("解释结果用于研究，不代表因果关系，也不能直接作为实盘依据。")
    lines.append("")

    lines.append("## 2026 年特征重要性")
    lines.append("")
    if importance.empty:
        lines.append("2026 年可解释样本太少，暂时无法计算稳定的特征重要性。")
    else:
        first = importance.iloc[0]
        lines.append(
            f"样本数量：{int(first['sample_count'])}；评分方式：{first['score_type']}；"
            f"基础分数：{float(first['base_score']):.4f}；模型：{first['model_name']}。"
        )
        lines.append("")
        lines.append("| 排名 | 特征 | 重要性 | 方向相关性 |")
        lines.append("|---:|---|---:|---:|")
        for rank, row in enumerate(importance[importance["importance"] > 0].head(15).itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | `{row.feature}` | {row.importance:.6f} | {row.direction_corr_2026:.4f} |"
            )
        if (importance["importance"] > 0).sum() == 0:
            lines.append("| - | 暂无正向稳定重要特征 | 0.000000 | 0.0000 |")
        lines.append("")
        lines.append("### 重要性和方向相关性怎么理解")
        lines.append("")
        lines.append("- `重要性` 回答的是：打乱这个特征后，模型表现会不会明显变差。它用来判断这个特征有没有被模型有效利用。")
        lines.append("- `方向相关性` 回答的是：这个特征偏高时，在 2026 年样本里更常对应跑赢还是跑输。它用来粗略判断“偏高更有利”还是“偏低更有利”。")
        lines.append("- `方向相关性 > 0`，粗略表示该特征偏高时更常对应 `label=1`，也就是未来 20 个交易日跑赢 AI 股票池平均收益。")
        lines.append("- `方向相关性 < 0`，粗略表示该特征偏低时更常对应 `label=1`。")
        lines.append("- `方向相关性 = 0` 或接近 0，不等于这个特征没用，只表示它没有明显的单向线性关系。它可能存在“太低不好、适中最好、太高也不好”的区间效应，或需要和其他特征组合才有意义。")
        lines.append("- 判断“打乱这个值有没有关系”，主要看 `重要性`；判断“这个值越高越好还是越低越好”，才看 `方向相关性`。")
        lines.append("- 如果 `重要性低` 且 `方向相关性接近 0`，才更接近说明这个特征在当前 2026 年样本里暂时帮助不大。")
        lines.append("")
        lines.append("### 当前特征字段中文解释")
        lines.append("")
        lines.append("- `ret_5`：过去 5 个交易日的涨跌幅。例如 5 天前收盘价 10 元，今天收盘价 11 元，`ret_5 = 10%`。")
        lines.append("- `ret_20`：过去 20 个交易日的涨跌幅，差不多对应 1 个月表现。")
        lines.append("- `ret_60`：过去 60 个交易日的涨跌幅，差不多对应 3 个月表现。")
        lines.append("- `ret_120`：过去 120 个交易日的涨跌幅，差不多对应半年表现。")
        lines.append("- `volume_chg_20`：最近 20 天平均成交量，相比再往前 20 天平均成交量的变化。它用来看成交量有没有明显放大或缩小。")
        lines.append("- `amount_chg_20`：最近 20 天平均成交额，相比再往前 20 天平均成交额的变化。成交额比成交量更接近资金参与强度。")
        lines.append("- `ma_20_gap`：当前收盘价相对 20 日均线的偏离程度。如果是正数，说明股价在 20 日均线上方；如果太高，可能代表短期涨得较快。")
        lines.append("- `ma_60_gap`：当前收盘价相对 60 日均线的偏离程度，它更偏中期趋势判断。")
        lines.append("- `vol_20`：过去 20 个交易日的日收益率波动率。数值越高，说明最近股价波动越大。")
        lines.append("- `vol_60`：过去 60 个交易日的日收益率波动率，比 `vol_20` 更偏中期风险水平。")
        lines.append("- `drawdown_60`：过去 60 个交易日里的最大回撤。比如最近 60 天最高点到最低点最多跌了 25%，这个值大约就是 -25%，用来看最近一段时间跌得深不深。")
        lines.append("- `high_60_breakout`：是否突破过去 60 个交易日的新高。通常是 0 或 1，1 表示今天收盘价突破了此前 60 日高点，0 表示没有。")
        lines.append("- `high_120_breakout`：是否突破过去 120 个交易日的新高，比 `high_60_breakout` 更偏中长期突破。")
        lines.append("- `relative_strength_20`：个股过去 20 日收益率，减去 AI 股票池平均 20 日收益率。如果为正，说明这只股票最近 20 天跑赢了 AI 股票池平均水平。")
        lines.append("- `relative_strength_60`：个股过去 60 日收益率，减去 AI 股票池平均 60 日收益率。如果为正，说明这只股票最近 60 天在 AI 股票池内部相对更强。")
    lines.append("")

    lines.append("## 最新一期前 20 名的简化原因")
    lines.append("")
    if latest_explain.empty:
        lines.append("暂无足够数据生成单股解释。")
    else:
        for rank, row in enumerate(latest_explain.itertuples(index=False), start=1):
            lines.append(
                f"{rank}. {row.stock_code} {row.stock_name}，p_outperform={float(row.p_outperform):.2%}"
            )
            lines.append(f"   - {row.reason_1}")
            lines.append(f"   - {row.reason_2}")
            lines.append(f"   - {row.reason_3}")
    lines.append("")

    lines.append("## 怎么阅读")
    lines.append("")
    lines.append("- `重要性` 越高，表示打乱这个特征后，模型在 2026 年样本上的表现下降越多。")
    lines.append("- `方向相关性` 大于 0，表示该特征偏高时在 2026 年样本里更常对应跑赢；小于 0 则相反。")
    lines.append("- 单股原因是简化解释，不是 SHAP，也不是因果结论。")
    lines.append("- 如果某个规则符合你的产业理解，再考虑把它发展成新特征或新约束。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_2026_explain_outputs(
    model: Pipeline,
    trainable: pd.DataFrame,
    latest: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    random_state: int,
) -> None:
    """生成 2026 年解释文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance = build_2026_feature_importance(
        model=model,
        data=trainable,
        feature_cols=feature_cols,
        model_name=model_name,
        random_state=random_state,
    )
    importance_path = OUTPUT_DIR / "feature_importance_2026.csv"
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    latest_explain = build_latest_explain_table(
        latest=latest,
        data=trainable,
        importance=importance,
        feature_cols=feature_cols,
        top_n=20,
    )
    latest_explain_path = OUTPUT_DIR / "latest_explain_2026.csv"
    latest_explain.to_csv(latest_explain_path, index=False, encoding="utf-8-sig")

    report_path = OUTPUT_DIR / "model_explain_report.md"
    write_2026_explain_report(importance, latest_explain, report_path)
    print(f"2026 年特征重要性已保存：{importance_path}")
    print(f"模型解释报告已保存：{report_path}")
