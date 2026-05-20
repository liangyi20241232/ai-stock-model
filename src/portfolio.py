from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_settings() -> dict[str, Any]:
    """读取仓位参数。"""
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_p_score(p_outperform: pd.Series, entry_threshold: float, full_confidence: float, gamma: float) -> pd.Series:
    """把跑赢概率映射成 0 到 1 的仓位分数。"""
    if full_confidence <= entry_threshold:
        raise ValueError("full_confidence 必须大于 entry_threshold")

    raw = (p_outperform - entry_threshold) / (full_confidence - entry_threshold)
    score = np.maximum(0, raw) ** gamma
    return pd.Series(np.minimum(score, 1), index=p_outperform.index)


def _allocate_one_layer(df: pd.DataFrame, budget: float, single_cap: float) -> pd.Series:
    """在一个层级内部按 p_score 归一化分配权重，并套用单只上限。"""
    if df.empty or budget <= 0:
        return pd.Series(0.0, index=df.index)

    positive = df["p_score"].clip(lower=0)
    if positive.sum() <= 0:
        return pd.Series(0.0, index=df.index)

    weights = positive / positive.sum() * budget
    weights = weights.clip(upper=single_cap)

    # 单只上限可能让实际总仓位低于预算；第一版不做复杂二次填充，保持保守。
    return weights


def _allocate_by_date(day_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """对单个信号日分别计算核心层和卫星层仓位。"""
    day_df = day_df.copy()

    entry_threshold = float(cfg["entry_threshold"])
    full_confidence = float(cfg["full_confidence"])
    gamma = float(cfg["probability_gamma"])
    max_total_weight = float(cfg["max_total_weight"])
    core_budget = max_total_weight * float(cfg["core_budget_ratio"])
    satellite_budget = max_total_weight * float(cfg["satellite_budget_ratio"])
    core_single_cap = float(cfg["core_single_cap"])
    satellite_single_cap = float(cfg["satellite_single_cap"])

    day_df["p_score"] = calculate_p_score(day_df["p_outperform"], entry_threshold, full_confidence, gamma)
    day_df["target_weight"] = 0.0

    core_mask = day_df["layer"] == "core_candidate"
    satellite_mask = day_df["layer"] == "satellite_candidate"
    day_df.loc[core_mask, "target_weight"] = _allocate_one_layer(
        day_df.loc[core_mask], core_budget, core_single_cap
    )
    day_df.loc[satellite_mask, "target_weight"] = _allocate_one_layer(
        day_df.loc[satellite_mask], satellite_budget, satellite_single_cap
    )

    total_weight = day_df["target_weight"].sum()
    if total_weight > max_total_weight:
        day_df["target_weight"] = day_df["target_weight"] / total_weight * max_total_weight

    return day_df


def generate_target_weights() -> pd.DataFrame:
    """根据 p_outperform 和 settings.yaml 逐个信号日生成建议目标仓位。"""
    settings = load_settings()
    cfg = settings["portfolio"]

    predictions_path = OUTPUT_DIR / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError("找不到 output/predictions.csv，请先运行：python main.py train")

    df = pd.read_csv(predictions_path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns.*")
        df = (
            df.groupby("date", group_keys=False)
            .apply(_allocate_by_date, cfg=cfg)
            .reset_index(drop=True)
        )

    output = df[
        ["date", "stock_code", "stock_name", "layer", "p_outperform", "p_score", "target_weight"]
    ].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output = output.sort_values(["date", "target_weight"], ascending=[True, False]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "target_weights.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"建议仓位已保存：{output_path}")
    return output


if __name__ == "__main__":
    generate_target_weights()
