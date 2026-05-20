#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "没有找到 .venv，请先运行：bash install_deps.sh"
  exit 1
fi

source .venv/bin/activate
export MPLCONFIGDIR="$PWD/.matplotlib_cache"

python main.py run-all
