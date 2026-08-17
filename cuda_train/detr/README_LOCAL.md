# 官方 DETR 训练入口

当前目录是官方 DETR 仓库源码。训练使用官方 `main.py`、`engine.py`、`datasets/coco.py` 和官方增强策略；本地只增加了 ResNet-50 权重加载参数。

## 需要下载的数据

官方代码要求标准 COCO 2017 目录，不支持当前 Hugging Face Parquet 目录。请在网络正常的机器下载以下官方文件，再复制到本机：

    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip

解压后目录必须是：

    /data/xumengying/models_and_datasets/coco2017/
      train2017/
      val2017/
      annotations/instances_train2017.json
      annotations/instances_val2017.json

不要在当前机器自动执行下载；下载好并解压后告诉我。

## ResNet-50 初始化

官方 DETR 检测器随机初始化，只有 ResNet-50 backbone 从本地 ImageNet 预训练模型加载：

    /data/xumengying/models_and_datasets/resnet-50/

脚本会把 Hugging Face ResNet-50 state dict 映射为官方 torchvision ResNet-50 backbone，并严格校验所有 backbone 参数都已加载。

## 单卡

    cd /data/xumengying/model-verify/cuda_train/detr
    source ../venv-cuda-py312/bin/activate
    python main.py \
      --coco_path /data/xumengying/models_and_datasets/coco2017 \
      --backbone_weights /data/xumengying/models_and_datasets/resnet-50 \
      --output_dir outputs/detr_resnet50 \
      --num_workers 8

默认官方配置为 300 epochs，第 200 epoch 学习率下降。

## 多卡

前台启动，不使用 nohup：

    torchrun --standalone --nproc_per_node=8 main.py \
      --coco_path /data/xumengying/models_and_datasets/coco2017 \
      --backbone_weights /data/xumengying/models_and_datasets/resnet-50 \
      --output_dir outputs/detr_resnet50_ddp \
      --num_workers 8

官方脚本只由主进程保存 checkpoint 和 log。训练结束后可用 `--eval --resume outputs/detr_resnet50_ddp/checkpoint.pth` 评测。

## 查看评测结果和 `latest.pth`

评测时建议将终端输出同时保存下来：

    torchrun --standalone --nproc_per_node=8 main.py \
      --coco_path /data/xumengying/models_and_datasets/coco2017 \
      --backbone_weights /data/xumengying/models_and_datasets/resnet-50 \
      --output_dir outputs/detr_resnet50_ddp/eval \
      --num_workers 8 \
      --eval \
      --resume outputs/detr_resnet50_ddp/checkpoint.pth \
      2>&1 | tee outputs/detr_resnet50_ddp/eval.log

## 日志

启动脚本使用 Python 非缓冲模式，并显式写入：

    outputs/detr_resnet50/train.log

日志第一条是 initialization 记录，包含：

- backbone 权重路径
- 成功加载的 backbone 参数数量
- backbone 是否加载成功
- 检测器是否随机初始化
- epochs 和 lr_drop

官方每个 epoch 的训练/验证指标仍写入 `outputs/detr_resnet50/log.txt`。多卡时只有 rank 0 写日志。
