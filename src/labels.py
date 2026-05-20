from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_settings() -> dict[str, Any]:
    """读取配置文件，标签周期等参数从 settings.yaml 获取。"""
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_future_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """计算单只股票未来收益。

    future_ret_20 = T+20 收盘价 / T+1 开盘价 - 1。
    这个字段是标签的一部分，可以使用未来数据；但它不会被放进特征列表里训练。
    """
    df = df.sort_values("date").copy()
    df[f"future_ret_{horizon}"] = df["close"].shift(-horizon) / df["open"].shift(-1) - 1
    return df


def build_dataset() -> pd.DataFrame:
    """把特征和标签合并成训练集，并保存到 data/processed/dataset.parquet。"""
    settings = load_settings()
    horizon = int(settings["model"].get("prediction_horizon", 20))

    feature_path = PROCESSED_DIR / "features.parquet"
    if feature_path.exists():
        features = pd.read_parquet(feature_path)
    else:
        features = build_features()

    features["date"] = pd.to_datetime(features["date"])
    features["stock_code"] = features["stock_code"].astype(str).str.zfill(6)

    dataset = (
        features.groupby("stock_code", group_keys=False)
        .apply(_add_future_return, horizon=horizon)
        .reset_index(drop=True)
    )

    future_ret_col = f"future_ret_{horizon}"
    ai_ret_col = f"future_ai_ret_{horizon}"

    # 同一信号日，AI 股票池所有股票未来收益的等权平均，作为“跑赢”基准。
    ai_future_mean = (
        dataset.groupby("date")[future_ret_col]
        .mean()
        .rename(ai_ret_col)
        .reset_index()
    )
    dataset = dataset.merge(ai_future_mean, on="date", how="left")
    dataset["future_ret_20"] = dataset[future_ret_col]
    dataset["future_ai_ret_20"] = dataset[ai_ret_col]
    dataset["label"] = (dataset[future_ret_col] > dataset[ai_ret_col]).astype("float")

    # 没有足够未来数据的最新日期不能用于训练，但仍会保留在 features.parquet 里供预测使用。
    dataset.loc[dataset[future_ret_col].isna() | dataset[ai_ret_col].isna(), "label"] = pd.NA

    output_path = PROCESSED_DIR / "dataset.parquet"
    dataset.to_parquet(output_path, index=False)
    print(f"训练数据已保存：{output_path}")
    return dataset


if __name__ == "__main__":
    build_dataset()
