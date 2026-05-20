#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

LIGHTGBM_LIB=".venv/lib/python3.9/site-packages/lightgbm/lib/lib_lightgbm.dylib"
SKLEARN_LIBOMP_DIR="$PWD/.venv/lib/python3.9/site-packages/sklearn/.dylibs"

if [ ! -f "$LIGHTGBM_LIB" ]; then
  echo "找不到 LightGBM 动态库，请先运行：bash install_deps.sh"
  exit 1
fi

if [ ! -f "$SKLEARN_LIBOMP_DIR/libomp.dylib" ]; then
  echo "找不到 sklearn 自带的 libomp.dylib，请确认 scikit-learn 已安装。"
  exit 1
fi

if otool -l "$LIGHTGBM_LIB" | grep -q "$SKLEARN_LIBOMP_DIR"; then
  echo "LightGBM 已经能找到本地 libomp。"
else
  cp "$LIGHTGBM_LIB" "$LIGHTGBM_LIB.bak"
  install_name_tool -add_rpath "$SKLEARN_LIBOMP_DIR" "$LIGHTGBM_LIB"
  echo "已为 LightGBM 添加本地 libomp 搜索路径。"
fi

.venv/bin/python - <<'PY'
import lightgbm
print("LightGBM 启用成功，版本：", lightgbm.__version__)
PY
