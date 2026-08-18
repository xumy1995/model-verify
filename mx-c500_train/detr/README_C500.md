# DETR 在 MetaX C500 上训练

本目录是 `cuda_train/detr` 的 C500 适配版本。C500 的 PyTorch 镜像通过
`torch.cuda` 提供设备接口，因此官方 DETR 的模型、数据集和分布式训练代码保持不变。

## 环境

使用 `maca-pytorch:3.8.1.2-torch2.10-py312` 镜像，并确认：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
python -m pip install -r requirements.txt
```

容器需要映射 `/dev/dri`、`/dev/mxcd`，并加入 `video` 组。数据应为标准 COCO 2017
目录，ResNet-50 权重目录可使用 `/mnt/afs/xumengying/models_and_datasets/resnet-50`。

## 冒烟测试

在小型 COCO 数据目录上运行：

```bash
python -u main.py --device cuda --coco_path /path/to/coco2017 \
  --backbone_weights /path/to/resnet-50 --output_dir outputs/smoke \
  --epochs 1 --batch_size 1 --num_workers 0
```

## 单卡训练

```bash
./run_detr.sh
```

## 多卡训练
```bash
torchrun --standalone --nproc_per_node=8 main.py --device cuda \
  --coco_path /mnt/afs/xumengying/models_and_datasets/coco2017 \
  --backbone_weights /mnt/afs/xumengying/models_and_datasets/resnet-50 \
  --output_dir outputs/detr_resnet50_ddp --num_workers 8
```

## 评测

```bash
torchrun --standalone --nproc_per_node=8 main.py --device cuda \
  --coco_path /mnt/afs/xumengying/models_and_datasets/coco2017 \
  --backbone_weights /mnt/afs/xumengying/models_and_datasets/resnet-50 \
  --output_dir outputs/detr_resnet50_ddp/eval \
  --num_workers 8 \
  --eval \
  --resume outputs/detr_resnet50_ddp/checkpoint.pth \
  2>&1 | tee outputs/detr_resnet50_ddp/eval.log
```