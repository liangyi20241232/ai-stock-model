from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_settings(settings_path: Path | None = None) -> dict[str, Any]:
    """读取 settings.yaml，所有策略参数都从这里进入程序，避免写死在代码里。"""
    path = settings_path or CONFIG_DIR / "settings.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ai_pool(pool_path: Path | None = None) -> pd.DataFrame:
    """读取 AI 算力与半导体股票池，并保留股票代码前导 0。"""
    path = pool_path or CONFIG_DIR / "ai_pool.csv"
    pool = pd.read_csv(path, dtype={"stock_code": str})
    pool["stock_code"] = pool["stock_code"].str.zfill(6)

    valid_layers = {"core_candidate", "satellite_candidate"}
    bad_layers = set(pool["layer"].dropna()) - valid_layers
    if bad_layers:
        raise ValueError(f"config/ai_pool.csv 中存在不支持的 layer: {bad_layers}")

    return pool


def _market_symbol(stock_code: str) -> str:
    """把 6 位股票代码转换成带市场前缀的代码，供腾讯/新浪接口使用。"""
    code = str(stock_code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _date_with_dash(date_text: str) -> str:
    """把 20210101 转成 2021-01-01，腾讯接口需要这种日期格式。"""
    text = str(date_text)
    if "-" in text:
        return text
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _normalize_akshare_columns(df: pd.DataFrame, stock_code: str, stock_name: str, source: str) -> pd.DataFrame:
    """把 AKShare 不同接口返回的字段统一成项目内部使用的字段。"""
    column_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=column_map).copy()

    if source == "tencent":
        # 腾讯接口在 AKShare 里返回的 amount 字段实际更接近成交量字段。
        # 第一版用 close * volume * 100 近似成交额，后续可切换到更精确的数据源。
        if "volume" not in df.columns and "amount" in df.columns:
            df["volume"] = df["amount"]
        if "volume" in df.columns:
            df["amount"] = df["volume"] * df["close"] * 100

    needed = ["date", "open", "high", "low", "close"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"AKShare 返回数据缺少字段: {missing}")

    if "volume" not in df.columns:
        df["volume"] = pd.NA
    if "amount" not in df.columns:
        df["amount"] = pd.NA

    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = str(stock_code).zfill(6)
    df["stock_name"] = stock_name

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_one_stock(stock_code: str, stock_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """使用 AKShare 拉取单只 A 股日线行情，并自动尝试多个免费数据源。

    第一版使用前复权价格，方便处理分红送转造成的价格跳变。
    TODO: 后续可以把复权方式也放进 settings.yaml。
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("未安装 akshare，请先运行：pip install -r requirements.txt") from exc

    code = str(stock_code).zfill(6)
    errors: list[str] = []
    data_sources = [
        (
            "eastmoney",
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                timeout=15,
            ),
        ),
        (
            "tencent",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=_market_symbol(code),
                start_date=_date_with_dash(start_date),
                end_date=_date_with_dash(end_date),
                adjust="qfq",
                timeout=15,
            ),
        ),
        (
            "sina",
            lambda: ak.stock_zh_a_daily(
                symbol=_market_symbol(code),
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ),
        ),
    ]

    for source, fetch_func in data_sources:
        for attempt in range(1, 3):
            try:
                raw = fetch_func()
                if raw is None or raw.empty:
                    raise ValueError("返回空数据")
                data = _normalize_akshare_columns(raw, code, stock_name, source)
                data.attrs["data_source"] = source
                return data
            except Exception as exc:  # noqa: BLE001 - 记录每个免费接口失败原因
                errors.append(f"{source} 第 {attempt} 次失败: {exc}")
                time.sleep(0.8)

    raise ValueError("；".join(errors))


def fetch_all_data() -> pd.DataFrame:
    """批量拉取股票池日线行情，单只失败不影响其他股票继续执行。"""
    settings = load_settings()
    pool = load_ai_pool()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start_date = str(settings["data"].get("start_date", "20210101"))
    end_date = str(settings["data"].get("end_date", "") or datetime.today().strftime("%Y%m%d"))

    logs: list[dict[str, str]] = []
    for row in tqdm(pool.itertuples(index=False), total=len(pool), desc="拉取日线行情"):
        stock_code = str(row.stock_code).zfill(6)
        stock_name = str(row.stock_name)
        try:
            prices = fetch_one_stock(stock_code, stock_name, start_date, end_date)
            prices.to_csv(RAW_DIR / f"{stock_code}.csv", index=False, encoding="utf-8-sig")
            data_source = str(prices.attrs.get("data_source", "unknown"))
            logs.append(
                {
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "status": "success",
                    "rows": str(len(prices)),
                    "data_source": data_source,
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - 这里需要记录单只股票的所有失败原因
            logs.append(
                {
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "status": "failed",
                    "rows": "0",
                    "data_source": "",
                    "message": str(exc),
                }
            )
            print(f"[提醒] {stock_code} {stock_name} 拉取失败：{exc}")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(RAW_DIR / "fetch_log.csv", index=False, encoding="utf-8-sig")
    print(f"行情拉取完成，日志已保存：{RAW_DIR / 'fetch_log.csv'}")
    return log_df


def load_raw_prices() -> pd.DataFrame:
    """读取 data/raw/ 中已经保存的本地日线行情。"""
    files = sorted(RAW_DIR.glob("*.csv"))
    files = [path for path in files if path.name != "fetch_log.csv"]
    if not files:
        raise FileNotFoundError("data/raw/ 里没有行情文件，请先运行：python main.py fetch-data")

    frames = []
    for path in files:
        df = pd.read_csv(path, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].str.zfill(6)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["stock_code", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    fetch_all_data()
