#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn akshare pyarrow matplotlib pyyaml tqdm

echo ""
echo "基础依赖安装完成。现在尝试安装 LightGBM..."
if pip install lightgbm; then
  echo "LightGBM 安装完成。"
else
  echo "LightGBM 安装失败，但项目仍可运行，会自动使用 RandomForestClassifier。"
fi

echo ""
echo "依赖安装流程结束。"
