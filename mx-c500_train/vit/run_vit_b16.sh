#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" vit/train_vit_b16.py "$@"
