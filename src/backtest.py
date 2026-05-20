from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_raw_prices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"


def load_settings() -> dict[str, Any]:
    """读取回测参数。"""
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _max_drawdown(equity: pd.Series) -> float:
    """计算净值曲线最大回撤。"""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _annualized_return(equity: pd.Series) -> float:
    """按 252 个交易日估算年化收益。"""
    if len(equity) < 2:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / 252
    if years <= 0:
        return np.nan
    return float((1 + total_return) ** (1 / years) - 1)


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    """计算不含无风险利率的简化夏普比率。"""
    if daily_returns.std() == 0 or daily_returns.dropna().empty:
        return np.nan
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))


def _monthly_last_signal_dates(weights: pd.DataFrame) -> list[pd.Timestamp]:
    """取每月最后一个有预测和仓位的交易日作为调仓信号日。"""
    signal_dates = (
        weights[["date"]]
        .drop_duplicates()
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month")["date"]
        .max()
        .sort_values()
        .tolist()
    )
    return signal_dates


def _next_trading_day(all_dates: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp | None:
    """找到信号日之后的下一个交易日，用于模拟 T+1 开盘买入。"""
    future_dates = all_dates[all_dates > signal_date]
    if len(future_dates) == 0:
        return None
    return pd.Timestamp(future_dates[0])


def run_backtest() -> pd.DataFrame:
    """运行月度调仓简化回测。

    限制说明：
    1. 不处理涨停买不进、跌停卖不出。
    2. 不做盘中成交模拟，只用 T+1 开盘买入、日收盘估值的简化方式。
    3. 历史预测来自一次时间切分后的测试期模型，不是逐月滚动重新训练。
    """
    settings = load_settings()
    cost = float(settings["backtest"].get("transaction_cost", 0.001))

    weights_path = OUTPUT_DIR / "target_weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError("找不到 output/target_weights.csv，请先运行：python main.py portfolio")

    weights = pd.read_csv(weights_path, dtype={"stock_code": str})
    weights["stock_code"] = weights["stock_code"].str.zfill(6)
    weights["date"] = pd.to_datetime(weights["date"])

    prices = load_raw_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)

    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    signal_dates = _monthly_last_signal_dates(weights)
    rebalance_rows = []
    for signal_date in signal_dates:
        trade_date = _next_trading_day(all_dates, signal_date)
        if trade_date is not None:
            rebalance_rows.append({"signal_date": signal_date, "trade_date": trade_date})

    if len(rebalance_rows) < 2:
        raise ValueError("可用调仓次数太少，无法回测。请确认 predictions.csv 包含多个历史月份。")

    rebalance = pd.DataFrame(rebalance_rows)
    open_px = prices.pivot(index="date", columns="stock_code", values="open").sort_index()
    close_px = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    stock_codes = sorted(set(weights["stock_code"]) & set(close_px.columns))

    daily_records = []
    previous_weights = pd.Series(0.0, index=stock_codes)

    for i, row in rebalance.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        trade_date = pd.Timestamp(row["trade_date"])
        next_trade_date = (
            pd.Timestamp(rebalance.iloc[i + 1]["trade_date"]) if i + 1 < len(rebalance) else all_dates[-1] + pd.Timedelta(days=1)
        )

        period_dates = all_dates[(all_dates >= trade_date) & (all_dates < next_trade_date)]
        if len(period_dates) == 0:
            continue

        day_weights = (
            weights[weights["date"] == signal_date]
            .set_index("stock_code")["target_weight"]
            .reindex(stock_codes)
            .fillna(0.0)
        )
        turnover = float((day_weights - previous_weights).abs().sum())
        previous_weights = day_weights.copy()

        for j, day in enumerate(period_dates):
            if day not in close_px.index:
                continue

            if j == 0:
                stock_ret = close_px.loc[day, stock_codes] / open_px.loc[day, stock_codes] - 1
            else:
                prev_day = period_dates[j - 1]
                stock_ret = close_px.loc[day, stock_codes] / close_px.loc[prev_day, stock_codes] - 1

            stock_ret = stock_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            portfolio_ret = float((day_weights * stock_ret).sum())
            benchmark_ret = float(stock_ret.mean())
            if j == 0:
                portfolio_ret -= turnover * cost

            daily_records.append(
                {
                    "date": day,
                    "signal_date": signal_date,
                    "turnover": turnover if j == 0 else 0.0,
                    "portfolio_return": portfolio_ret,
                    "benchmark_return": benchmark_ret,
                }
            )

    if not daily_records:
        raise ValueError("没有生成任何回测日收益，请检查行情数据和调仓日期。")

    daily = pd.DataFrame(daily_records).sort_values("date")
    daily["portfolio_equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()
    daily["excess_equity"] = daily["portfolio_equity"] / daily["benchmark_equity"]

    monthly_returns = daily.set_index("date")["portfolio_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    total_return = float(daily["portfolio_equity"].iloc[-1] - 1)
    benchmark_total_return = float(daily["benchmark_equity"].iloc[-1] - 1)

    metrics = pd.DataFrame(
        [
            {"metric": "annualized_return", "value": _annualized_return(daily["portfolio_equity"])},
            {"metric": "max_drawdown", "value": _max_drawdown(daily["portfolio_equity"])},
            {"metric": "win_rate", "value": float((daily["portfolio_return"] > 0).mean())},
            {"metric": "monthly_win_rate", "value": float((monthly_returns > 0).mean())},
            {"metric": "sharpe_ratio", "value": _sharpe_ratio(daily["portfolio_return"])},
            {"metric": "turnover", "value": float(daily["turnover"].sum())},
            {"metric": "total_return", "value": total_return},
            {"metric": "benchmark_total_return", "value": benchmark_total_return},
            {"metric": "excess_return_vs_ai_equal_weight", "value": total_return - benchmark_total_return},
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "backtest_result.csv"
    equity_path = OUTPUT_DIR / "equity_curve.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    daily.to_csv(equity_path, index=False, encoding="utf-8-sig")
    print(f"回测指标已保存：{metrics_path}")
    print(f"净值曲线已保存：{equity_path}")
    return metrics


if __name__ == "__main__":
    run_backtest()
