from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_ai_pool, load_raw_prices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


FEATURE_COLUMNS = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "volume_chg_20",
    "amount_chg_20",
    "ma_20_gap",
    "ma_60_gap",
    "vol_20",
    "vol_60",
    "drawdown_60",
    "high_60_breakout",
    "high_120_breakout",
    "relative_strength_20",
    "relative_strength_60",
]


def _rolling_max_drawdown(values: np.ndarray) -> float:
    """计算一个滚动窗口内的最大回撤，结果通常是 0 或负数。"""
    if len(values) == 0 or np.isnan(values).any():
        return np.nan
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1
    return float(drawdowns.min())


def _build_one_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票生成只使用 T 日及以前数据的日线特征。"""
    df = df.sort_values("date").copy()
    daily_ret = df["close"].pct_change()

    # ret_5/20/60/120：过去 N 个交易日收益率，只使用当前日及以前收盘价。
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)
    df["ret_120"] = df["close"].pct_change(120)

    # volume_chg_20：最近 20 日均量相对更早 20 日均量的变化。
    volume_ma_20 = df["volume"].rolling(20).mean()
    df["volume_chg_20"] = volume_ma_20 / volume_ma_20.shift(20) - 1

    # amount_chg_20：最近 20 日均成交额相对更早 20 日均成交额的变化。
    amount_ma_20 = df["amount"].rolling(20).mean()
    df["amount_chg_20"] = amount_ma_20 / amount_ma_20.shift(20) - 1

    # ma_20_gap/ma_60_gap：收盘价相对均线的偏离程度。
    ma_20 = df["close"].rolling(20).mean()
    ma_60 = df["close"].rolling(60).mean()
    df["ma_20_gap"] = df["close"] / ma_20 - 1
    df["ma_60_gap"] = df["close"] / ma_60 - 1

    # vol_20/vol_60：过去 N 日日收益率波动率。
    df["vol_20"] = daily_ret.rolling(20).std()
    df["vol_60"] = daily_ret.rolling(60).std()

    # drawdown_60：过去 60 日窗口内最大回撤。
    df["drawdown_60"] = df["close"].rolling(60).apply(_rolling_max_drawdown, raw=True)

    # high_60/120_breakout：当前收盘价是否突破此前 N 日最高价。
    # 这里用 shift(1) 排除当天，避免“当天自己创造的新高”造成含义不清。
    df["high_60_breakout"] = (df["close"] > df["high"].rolling(60).max().shift(1)).astype(int)
    df["high_120_breakout"] = (df["close"] > df["high"].rolling(120).max().shift(1)).astype(int)

    return df


def build_features() -> pd.DataFrame:
    """生成全股票池特征，并保存到 data/processed/features.parquet。"""
    prices = load_raw_prices()
    pool = load_ai_pool()[["stock_code", "theme_1", "theme_2", "layer"]]
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    features = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_build_one_stock_features)
        .reset_index(drop=True)
    )

    # AI 股票池平均 20/60 日收益，用于计算相对强弱。
    ai_mean = (
        features.groupby("date")[["ret_20", "ret_60"]]
        .mean()
        .rename(columns={"ret_20": "ai_pool_ret_20", "ret_60": "ai_pool_ret_60"})
        .reset_index()
    )
    features = features.merge(ai_mean, on="date", how="left")

    # relative_strength_20/60：个股过去收益率减去 AI 股票池平均收益率。
    features["relative_strength_20"] = features["ret_20"] - features["ai_pool_ret_20"]
    features["relative_strength_60"] = features["ret_60"] - features["ai_pool_ret_60"]

    features = features.merge(pool, on="stock_code", how="left")
    features = features.sort_values(["date", "stock_code"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(output_path, index=False)
    print(f"特征已保存：{output_path}")
    return features


if __name__ == "__main__":
    build_features()
