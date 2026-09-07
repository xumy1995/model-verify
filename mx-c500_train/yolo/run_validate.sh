#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -u validate_yolo.py "$@"
