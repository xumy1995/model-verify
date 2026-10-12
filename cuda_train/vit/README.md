# CUDA ViT-B/16 训练

本目录使用与 `mx-c500_train/vit/train_vit_b16.py` 相同的 ViT-B/16 模型、数据增强、优化器、学习率、EMA、AMP 和训练参数，在 8 张 NVIDIA A100 上复现 ImageNet-1K 训练。

## 环境与数据

```bash
cd /data/xumengying/model-verify/cuda_train
source venv-cuda-py312/bin/activate
```

默认数据路径为 `/data/xumengying/models_and_datasets/ILSVRC`，需要包含：

```text
Data/CLS-LOC/train/<synset>/*.JPEG
Data/CLS-LOC/val/*.JPEG
ImageSets/CLS-LOC/val.txt
Annotations/CLS-LOC/val/*.xml
```

数据完整性应为 1000 个类别、1,281,167 张训练图、50,000 张验证图和 50,000 行验证列表。

## 配方

默认使用 8 卡 DDP、每卡 batch 512（有效全局 batch 4096）、300 epochs、AdamW、lr 0.003、weight decay 0.3、30 epoch warmup、CosineAnnealing、AMP、label smoothing 0.11、MixUp 0.2、CutMix 1.0、RandAugment、3 次 repeated augmentation、梯度裁剪 1.0 和 EMA。

与 MX-C500 版本一致，默认每个 rank 使用 2 个 DataLoader worker 和 1 个 batch 预取。A100 试跑已确认原始每卡 batch 512 可用。

## 冒烟测试

```bash
mkdir -p vit/logs vit/checkpoints
NPROC_PER_NODE=1 ./vit/run_vit_b16.sh --epochs 1 --train-batches 1 --val-batches 1 \
  --output-dir vit/checkpoints/probe --log-file vit/logs/probe.jsonl
```

## 完整训练

```bash
mkdir -p vit/logs vit/checkpoints
./vit/run_vit_b16.sh 2>&1 | tee vit/logs/vit_b16_v1_console.log
```

训练过程写入 `vit/logs/vit_b16_v1_ddp.jsonl`，rank 0 保存 `vit/checkpoints/vit_b16_v1_ddp/last.pth` 和按 EMA Top-1 选择的 `best.pth`。

断点续训：

```bash
./vit/run_vit_b16.sh --resume vit/checkpoints/vit_b16_v1_ddp/last.pth
```

## 独立评测

```bash
./vit/run_vit_b16.sh --help
python vit/eval_vit_b16.py \
  --checkpoint vit/checkpoints/vit_b16_v1_ddp/best.pth \
  --weights ema --device cuda --batch-size 256 --workers 8 \
  | tee vit/logs/eval_vit_b16_v1_best_ema.log
```

评测输出 Top-1/Top-5、端到端吞吐、纯模型吞吐和平均推理延迟；使用 `--weights model` 可评测非 EMA 权重，使用 `--fp16` 可评测 FP16 推理。

## 复现结果

CUDA 训练已完成 300 epochs；最佳 EMA checkpoint 出现在第 280 epoch，训练日志中
记录的最佳 EMA Top-1/Top-5 为 80.934%/95.248%。训练最后阶段曾在 NCCL 进程组
清理时报告 CUDA OOM，但第 300 epoch 的 checkpoint 已成功保存，`last.pth` 为
epoch 300，`best.pth` 为 epoch 280，不影响结果使用。

独立评测使用 `best.pth` 的 EMA 权重、单张 A100、FP32、batch 256、2 个 workers，
完整评测 50,000 张验证图。结果如下：

| 指标 | CUDA A100 | MX-C500 |
|---|---:|---:|
| Top-1 | 80.932% | 80.954% |
| Top-5 | 95.246% | 95.348% |
| 纯模型吞吐 | 1972.57 samples/s | 1001.44 samples/s |
| 平均推理延迟 | 0.507 ms/sample | 0.999 ms/sample |

结果日志：

- [CUDA 训练控制台日志](logs/vit_b16_v1_console.log)
- [CUDA 训练 JSONL 日志](logs/vit_b16_v1_ddp.jsonl)
- [CUDA best EMA 独立评测日志](logs/eval_vit_b16_v1_best_ema.log)
- [MX-C500 best EMA 独立评测日志](../../mx-c500_train/vit/logs/eval_vit_b16_v1_best_ema.log)
