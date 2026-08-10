#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# source venv-cuda-py312/bin/activate

python resnet/train_resnet50.py \
  --recipe v2 \
  --data /mnt/afs/xumengying/models_and_datasets/ILSVRC \
  --output-dir resnet/checkpoints/resnet50_v2 \
  --log-file resnet/logs/resnet50_v2_train.log \
  "$@"
