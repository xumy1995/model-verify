#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p cuda_train/clip/results
cuda_train/venv-cuda-py312/bin/python -m cuda_train.clip.eval_imagenet "$@" \
  2>&1 | tee cuda_train/clip/results/imagenet_eval.log
