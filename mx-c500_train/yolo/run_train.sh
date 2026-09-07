#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -u train_yolo.py "$@"
