#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# C500's PyTorch runtime exposes the accelerator through torch.cuda.
python -u main.py \
  --device cuda \
  --coco_path /mnt/afs/xumengying/models_and_datasets/coco2017 \
  --backbone_weights /mnt/afs/xumengying/models_and_datasets/resnet-50 \
  --output_dir outputs/detr_resnet50 \
  --num_workers "${NUM_WORKERS:-8}" \
  "$@"
