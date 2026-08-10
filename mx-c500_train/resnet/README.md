# mx-c500 训练验证

本目录用于验证 ResNet-50 在 mx-c500 上的训练流程，并对齐 TorchVision 官方 ImageNet-1K 基线。

## 环境

使用此前测试验证时创建的docker环境:

     sudo docker exec -it maca-pytorch-test bash

训练脚本使用本目录已有的 torch、torchvision、Pillow 环境，不会自动下载数据或依赖。

## Kaggle 数据目录

当前训练脚本已经适配 Kaggle 下载的 ILSVRC 目录，不需要转换成 ImageFolder，也不需要重新解压：

    /mnt/afs/xumengying/models_and_datasets/ILSVRC/
      Data/CLS-LOC/train/<synset>/*.JPEG
      Data/CLS-LOC/val/*.JPEG
      ImageSets/CLS-LOC/val.txt
      ImageSets/CLS-LOC/train_cls.txt
      Annotations/CLS-LOC/val/*.xml

训练集类别目录直接由 torchvision.datasets.ImageFolder 读取。验证集图片是平铺的，脚本使用 ImageSets/CLS-LOC/val.txt 将图片名映射到 1-1000 类别标签。

确认数据完整性：

    find /mnt/afs/xumengying/models_and_datasets/ILSVRC/Data/CLS-LOC/train -mindepth 1 -maxdepth 1 -type d | wc -l
    find /mnt/afs/xumengying/models_and_datasets/ILSVRC/Data/CLS-LOC/train -type f | wc -l
    find /mnt/afs/xumengying/models_and_datasets/ILSVRC/Data/CLS-LOC/val -type f | wc -l
    wc -l /mnt/afs/xumengying/models_and_datasets/ILSVRC/ImageSets/CLS-LOC/val.txt

期望输出分别为：

    1000
    1281167
    50000
    50000

## ResNet-50 官方基线

| 配方 | TorchVision 权重 | 官方 Top-1 | 官方 Top-5 | 说明 |
|---|---|---|---|---|
| v1 | ResNet50_Weights.IMAGENET1K_V1 | 76.130 | 92.862 | 经典 90 epoch SGD StepLR 配方 |
| v2 | ResNet50_Weights.IMAGENET1K_V2 / default | 80.858 | 95.434 | TorchVision new recipe |

默认启动脚本使用 v2，对齐 TorchVision 当前默认 ResNet-50 基线。

## 冒烟测试

    ./resnet/run_resnet50_v2.sh --smoke

该命令只跑 1 个训练 batch 和 1 个验证 batch，用于确认数据读取、CUDA、AMP、日志和 checkpoint 写入链路正常。

## 完整基线训练

建议在当前终端前台启动，这样可以直接看到 batch 进度：

    ./resnet/run_resnet50_v2.sh --log-interval 10

脚本每 100 个 batch 输出一次进度，每个 epoch 结束输出一次验证结果。

多卡使用 torchrun 启动；batch-size 是每张卡的 batch size：

    torchrun --standalone --nproc_per_node=8 resnet/train_resnet50.py --recipe v2 --data /mnt/afs/xumengying/models_and_datasets/ILSVRC --output-dir resnet/checkpoints/resnet50_v2_ddp --log-file resnet/logs/resnet50_v2_ddp.log --batch-size 128 --workers 8 --log-interval 100

多卡时仅 rank 0 写日志和 checkpoint，训练集使用 DistributedSampler，验证指标会跨 GPU 汇总。停止训练请在启动 torchrun 的前台终端按 Ctrl-C。

输出文件：

- 训练日志：resnet/logs/resnet50_v2_train.log
- 最新 checkpoint：resnet/checkpoints/resnet50_v2/last.pth

断点续训：

    ./resnet/run_resnet50_v2.sh --resume resnet/checkpoints/resnet50_v2/last.pth

## 独立评测

对训练好的 checkpoint 进行精度和性能评测：

    python resnet/eval_resnet50.py \
      --checkpoint resnet/checkpoints/resnet50_v2_ddp/last.pth \
      --data /mnt/afs/xumengying/models_and_datasets/ILSVRC \
      --devices cuda \
      --batch-size 256 \
      --workers 8 \
      --warmup-batches 5 | tee resnet/logs/eval_resnet50_v2_ddp.log

默认使用 FP32。需要测试 CUDA FP16 时再额外添加 `--fp16`。

输出指标包括：

- Top-1 / Top-5 accuracy
- 端到端吞吐（包含数据搬运）
- 纯模型推理吞吐
- 平均纯推理延迟（毫秒/样本）
- 模型加载耗时

评测脚本默认读取训练脚本保存的 checkpoint 格式：`{"model": state_dict, ...}`。
