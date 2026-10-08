#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source venv-cuda-py312/bin/activate
mkdir -p yolo/logs

python yolo/train_yolo.py "$@"
