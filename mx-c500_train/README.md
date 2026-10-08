# mx-c500 训练

模型训练按模型分别放置：

- `resnet/`：ResNet-50 ImageNet 训练与验证
- `detr/`：DETR ResNet-50 COCO 训练与验证（见 `detr/README_C500.md`）
- `vit/`：ViT-B/16 ImageNet-1K 官方 recipe 训练与验证
- `yolo/`：YOLO26 COCO 训练与验证

在 MetaX C500 的 maca-pytorch 容器内执行训练；`mx-c500_train` 下没有 CUDA 虚拟环境。进入各模型目录的 README（DETR 见 `detr/README_C500.md`）查看数据、环境和训练命令。
