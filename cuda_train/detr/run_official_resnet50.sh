#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ../venv-cuda-py312/bin/activate

python -u main.py \
  --coco_path /data/xumengying/models_and_datasets/coco2017 \
  --backbone_weights /data/xumengying/models_and_datasets/resnet-50 \
  --output_dir outputs/detr_resnet50 \
  --log_file outputs/detr_resnet50/train.log \
  --num_workers 8 \
  "$@"
