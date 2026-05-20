from __future__ import annotations

import argparse
import sys


def build_features_and_labels() -> None:
    """生成特征和标签。"""
    from src.features import build_features
    from src.labels import build_dataset

    build_features()
    build_dataset()


def step_fetch_data() -> None:
    """拉取行情数据。"""
    from src.data_loader import fetch_all_data

    fetch_all_data()


def step_train() -> None:
    """训练模型并生成预测。"""
    from src.model import train_and_predict

    train_and_predict()


def step_portfolio() -> None:
    """生成建议仓位。"""
    from src.portfolio import generate_target_weights

    generate_target_weights()


def step_backtest() -> None:
    """运行简化回测。"""
    from src.backtest import run_backtest

    run_backtest()


def step_report() -> None:
    """生成 Markdown 报告。"""
    from src.report import generate_report

    generate_report()


def run_all() -> None:
    """按顺序运行完整闭环。"""
    steps = [
        ("拉取行情数据", step_fetch_data),
        ("生成特征和标签", build_features_and_labels),
        ("训练模型并生成预测", step_train),
        ("生成建议仓位", step_portfolio),
        ("运行简化回测", step_backtest),
        ("生成 Markdown 报告", step_report),
    ]
    for step_name, func in steps:
        print(f"\n========== {step_name} ==========")
        func()


def main() -> None:
    """命令行入口，适合新手一步一步运行。"""
    parser = argparse.ArgumentParser(description="中国 A 股 AI 算力与半导体股票池预测系统")
    parser.add_argument(
        "command",
        choices=["fetch-data", "build-features", "train", "portfolio", "backtest", "report", "run-all"],
        help="要执行的步骤",
    )
    args = parser.parse_args()

    command_map = {
        "fetch-data": ("拉取行情数据", step_fetch_data),
        "build-features": ("生成特征和标签", build_features_and_labels),
        "train": ("训练模型并生成预测", step_train),
        "portfolio": ("生成建议仓位", step_portfolio),
        "backtest": ("运行简化回测", step_backtest),
        "report": ("生成 Markdown 报告", step_report),
        "run-all": ("运行完整流程", run_all),
    }

    step_name, func = command_map[args.command]
    print(f"\n开始：{step_name}")
    try:
        func()
    except FileNotFoundError as exc:
        print(f"\n[文件缺失] {exc}")
        print("建议：按 README.md 的顺序运行，或直接运行 python main.py run-all。")
        sys.exit(1)
    except ImportError as exc:
        print(f"\n[依赖缺失] {exc}")
        print("建议：先运行 pip install -r requirements.txt。")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - 面向新手时需要把错误解释清楚
        print(f"\n[运行失败] {exc}")
        print("建议：把上面这段报错完整发给我，我会继续帮你修。")
        sys.exit(1)
    print(f"完成：{step_name}")


if __name__ == "__main__":
    main()
