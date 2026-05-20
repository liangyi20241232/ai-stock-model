from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explain import generate_2026_explain_outputs
from src.features import FEATURE_COLUMNS
from src.labels import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def load_settings() -> dict[str, Any]:
    """读取模型训练参数。"""
    with (CONFIG_DIR / "settings.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _time_split(dataset: pd.DataFrame, test_size_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按日期切分训练集和测试集，不能随机打乱，避免时间穿越。"""
    dates = np.array(sorted(dataset["date"].dropna().unique()))
    if len(dates) < 5:
        raise ValueError("可训练日期太少，请先拉取更长时间的数据。")

    split_idx = max(1, int(len(dates) * (1 - test_size_ratio)))
    split_idx = min(split_idx, len(dates) - 1)
    split_date = dates[split_idx]

    train_df = dataset[dataset["date"] < split_date].copy()
    test_df = dataset[dataset["date"] >= split_date].copy()
    return train_df, test_df


def _make_model(random_state: int):
    """优先使用 LightGBM；不可用时自动降级到随机森林，再降级到逻辑回归。"""
    try:
        from lightgbm import LGBMClassifier

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.03,
                        num_leaves=15,
                        random_state=random_state,
                        verbose=-1,
                    ),
                ),
            ]
        )
        return model, "LightGBM"
    except Exception as exc:  # noqa: BLE001 - LightGBM 在不同 Mac 环境可能导入失败
        print(f"[提醒] LightGBM 不可用，改用 RandomForestClassifier。原因：{exc}")

    try:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=20,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return model, "RandomForestClassifier"
    except Exception as exc:  # noqa: BLE001
        print(f"[提醒] RandomForestClassifier 不可用，改用 LogisticRegression。原因：{exc}")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    return model, "LogisticRegression"


def _score_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[str, float]:
    """优先输出 AUC；如果测试集只有单一类别，则输出 accuracy。"""
    if len(x_test) == 0:
        return "no_test_data", np.nan

    p = model.predict_proba(x_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    if y_test.nunique(dropna=True) >= 2:
        return "auc", float(roc_auc_score(y_test, p))
    return "accuracy", float(accuracy_score(y_test, pred))


def train_and_predict() -> pd.DataFrame:
    """训练分类模型，并输出最新一期 p_outperform 到 output/predictions.csv。"""
    settings = load_settings()
    test_size_ratio = float(settings["model"].get("test_size_ratio", 0.25))
    random_state = int(settings["model"].get("random_state", 42))

    dataset_path = PROCESSED_DIR / "dataset.parquet"
    if dataset_path.exists():
        dataset = pd.read_parquet(dataset_path)
    else:
        dataset = build_dataset()

    dataset["date"] = pd.to_datetime(dataset["date"])
    trainable = dataset.dropna(subset=["label"]).copy()
    feature_cols = [col for col in FEATURE_COLUMNS if col in trainable.columns]
    if not feature_cols:
        raise ValueError("没有找到可用特征，请先运行：python main.py build-features")

    trainable = trainable.dropna(subset=feature_cols, how="all")
    if trainable["label"].nunique(dropna=True) < 2:
        raise ValueError("训练标签只有一个类别，暂时无法训练分类模型。请拉取更长历史数据或扩充股票池。")

    train_df, test_df = _time_split(trainable, test_size_ratio)
    if train_df["label"].nunique(dropna=True) < 2:
        raise ValueError("训练集标签只有一个类别，暂时无法训练。请拉取更长历史数据或扩充股票池。")

    x_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    model, model_name = _make_model(random_state)
    try:
        model.fit(x_train, y_train)
    except Exception as exc:  # noqa: BLE001
        if model_name != "LogisticRegression":
            print(f"[提醒] {model_name} 训练失败，改用 LogisticRegression。原因：{exc}")
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
                ]
            )
            model_name = "LogisticRegression"
            model.fit(x_train, y_train)
        else:
            raise

    score_name, score_value = _score_model(model, x_test, y_test)

    # 先生成测试期历史预测，用于后续简化回测。
    historical = test_df.copy()
    historical["p_outperform"] = model.predict_proba(historical[feature_cols])[:, 1]

    # 最新预测使用 features.parquet 的最新日期，不要求有未来标签。
    latest_features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    latest_features["date"] = pd.to_datetime(latest_features["date"])
    latest_date = latest_features["date"].max()
    latest = latest_features[latest_features["date"] == latest_date].copy()
    latest = latest.dropna(subset=feature_cols, how="all")
    if latest.empty:
        raise ValueError("最新日期没有可用于预测的特征，请检查 data/processed/features.parquet。")

    latest["p_outperform"] = model.predict_proba(latest[feature_cols])[:, 1]
    generate_2026_explain_outputs(
        model=model,
        trainable=trainable,
        latest=latest,
        feature_cols=feature_cols,
        model_name=model_name,
        random_state=random_state,
    )

    output = pd.concat([historical, latest], ignore_index=True)
    output["model_name"] = model_name
    output["model_score"] = score_value
    output["model_score_type"] = score_name
    output = output.drop_duplicates(subset=["date", "stock_code"], keep="last")
    output = output[
        ["date", "stock_code", "stock_name", "layer", "p_outperform", "model_name", "model_score"]
    ].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output = output.sort_values(["date", "p_outperform"], ascending=[True, False]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "predictions.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已保存：{output_path}，模型：{model_name}，{score_name}={score_value:.4f}")
    return output


if __name__ == "__main__":
    train_and_predict()
