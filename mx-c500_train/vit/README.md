# ViT-B/16 在 MetaX C500 上训练

本目录使用 TorchVision `vit_b_16(weights=None)` 在 ImageNet-1K 上从零训练，配方对齐
`ViT_B_16_Weights.IMAGENET1K_V1` 的 modified DeiT reference recipe。官方精度为
Top-1 81.072%、Top-5 95.318%。C500 通过 `torch.cuda` 接口使用。

## 环境与数据

在宿主机进入已有容器：

```bash
sudo docker exec -it maca-pytorch-test bash
cd /workspace/model-verify/mx-c500_train
```

默认读取 `/mnt/afs/xumengying/models_and_datasets/ILSVRC`。脚本支持标准
`train/<class>`、`val/<class>` ImageFolder，也支持仓库现有的 Kaggle ILSVRC 布局：

```text
Data/CLS-LOC/train/<synset>/*.JPEG
Data/CLS-LOC/val/*.JPEG
ImageSets/CLS-LOC/val.txt
Annotations/CLS-LOC/val/*.xml
```

## 官方配方

默认使用 8 卡 DDP、每卡 batch 512（有效全局 batch 4096）、300 epochs、AdamW、
lr 0.003、weight decay 0.3、30 epoch 线性 warmup、CosineAnnealing、AMP、label
smoothing 0.11、MixUp 0.2、CutMix 1.0、RandAugment、3 次 repeated augmentation、
梯度裁剪 1.0 和 EMA。

默认每个 rank 使用 2 个 DataLoader worker 和 1 个 batch 的预取深度，以适配当前
容器的 16 GB `/dev/shm`；增加 worker 前应同步增大容器共享内存。

## 冒烟与完整训练

```bash
NPROC_PER_NODE=1 ./vit/run_vit_b16.sh --smoke \
  --output-dir vit/checkpoints/smoke_single --log-file vit/logs/smoke_single.jsonl

./vit/run_vit_b16.sh --smoke \
  --output-dir vit/checkpoints/smoke_ddp --log-file vit/logs/smoke_ddp.jsonl

./vit/run_vit_b16.sh 2>&1 | tee vit/logs/vit_b16_v1_console.log
```

若每卡 batch 512 实测显存不足，使用每卡 256、累积 2 步保持有效全局 batch 4096：

```bash
./vit/run_vit_b16.sh --batch-size 256 --accumulation-steps 2
```

断点续训：

```bash
./vit/run_vit_b16.sh --resume vit/checkpoints/vit_b16_v1_ddp/last.pth
```

训练过程写入 JSONL 日志，仅 rank 0 保存 `last.pth` 和按 EMA Top-1 选择的
`best.pth`。

## 独立评测

```bash
python vit/eval_vit_b16.py \
  --checkpoint vit/checkpoints/vit_b16_v1_ddp/best.pth \
  --weights ema --device cuda --batch-size 256
```

评测输出 Top-1/Top-5、端到端吞吐、纯推理吞吐和平均推理延迟。使用
`--weights model` 可评测非 EMA 权重，使用 `--fp16` 可评测 FP16 推理。
