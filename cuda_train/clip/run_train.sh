#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
DATA_DIR=/data/xumengying/models_and_datasets/cc3m_wds
PYTHON=cuda_train/venv-cuda-py312/bin/python
TORCHRUN=cuda_train/venv-cuda-py312/bin/torchrun
MODE=${1:-train}
case "$MODE" in
  train)
    NAME=cc3m_vit_b32_8xa100
    TRAIN_SHARDS="$DATA_DIR/cc3m-train-{0000..0575}.tar"
    TRAIN_SAMPLES=2905954
    EPOCHS=32
    WARMUP=2000
    VAL_SHARDS="$DATA_DIR/cc3m-validation-{0000..0015}.tar"
    VAL_SAMPLES=13443
    ;;
  smoke)
    NAME=cc3m_vit_b32_smoke
    TRAIN_SHARDS="$DATA_DIR/cc3m-train-{0000..0015}.tar"
    TRAIN_SAMPLES=4096
    EPOCHS=1
    WARMUP=2
    VAL_SHARDS="$DATA_DIR/cc3m-validation-0000.tar"
    VAL_SAMPLES=840
    ;;
  *) echo "Usage: $0 [train|smoke] [additional OpenCLIP options]" >&2; exit 2 ;;
esac
shift || true
test -f "$DATA_DIR/cc3m-train-0575.tar"
test -f "$DATA_DIR/cc3m-validation-0015.tar"
"$PYTHON" -c 'import open_clip, open_clip_train, webdataset'
exec "$TORCHRUN" --standalone --nproc_per_node=8 -m open_clip_train.main \
  --train-data "$TRAIN_SHARDS" \
  --val-data "$VAL_SHARDS" \
  --train-num-samples "$TRAIN_SAMPLES" --val-num-samples "$VAL_SAMPLES" \
  --dataset-type webdataset --model ViT-B-32 \
  --batch-size 64 --accum-freq 1 --workers 2 \
  --epochs "$EPOCHS" --warmup "$WARMUP" --lr 0.0005 \
  --beta1 0.9 --beta2 0.98 --eps 0.000001 --wd 0.2 \
  --precision amp_bf16 --seed 0 --val-frequency 1 \
  --log-every-n-steps 20 --save-frequency 1 \
  --logs cuda_train/clip/logs --name "$NAME" "$@"
