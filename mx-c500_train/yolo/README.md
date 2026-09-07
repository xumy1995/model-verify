# 在 MetaX C500 上训练 YOLO26

本目录用于在 MX-C500 镜像容器内使用 8 张卡从零训练 YOLO26。使用容器中已安装的 Ultralytics（项目评测环境为 8.4.115）和 maca-pytorch；通过逗号分隔的 `device` 选择设备，默认值为 `0,1,2,3,4,5,6,7`。

默认从官方起始权重 `yolo26n-objv1-150.pt` 开始训练，使用独立输出目录，不覆盖第一次训练结果。

## 进入 MX-C500 容器

宿主机启动容器（与你实际使用的命令一致）：

```bash
sudo docker run -it --name maca-pytorch-test \
  --device=/dev/dri --device=/dev/mxcd --group-add video \
  --shm-size=16g \
  -v /mnt/afs/xumengying/model-verify:/workspace/model-verify \
  -v /mnt/afs/xumengying/models_and_datasets:/mnt/afs/xumengying/models_and_datasets \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

进入容器后执行：

```bash
cd /workspace/model-verify/mx-c500_train/yolo
```

代码和数据分别通过上面的两个 `-v` 挂载；不要在宿主机直接执行训练脚本。若容器已创建，可使用 `sudo docker exec -it maca-pytorch-test bash` 重新进入。

## 数据准备

`--data` 指向 Ultralytics YOLO 格式的 YAML。COCO 示例默认使用：
`/mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml`。
YAML 至少应包含 `path`、`train`、`val` 和 `names`；标签为每行 `class x_center y_center width height`，坐标归一化到 0--1。

## 开始训练

```bash
cd /workspace/model-verify/mx-c500_train/yolo
bash run_train.sh --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26n-objv1-150.pt \
  --data /mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml \
  --epochs 100 --batch 128 --imgsz 640 --device 0,1,2,3,4,5,6,7 \
  --optimizer MuSGD --name yolo26n_official_recipe 2>&1 | tee logs/train_yolo26n_official.log
```

脚本已显式对齐官方 checkpoint 中记录的优化器、学习率、warmup、loss 权重、数据增强、MuSGD/YOLO26 专用参数、AMP 和 deterministic 配置；epoch 按官方 recipe 示例使用 100。

输出默认保存到 `mx-c500_train/yolo/runs/<name>/`，其中 `weights/best.pt` 是验证集表现最佳的权重。显存不足时降低 `--batch`；多卡可传 `--device 0,1`（按 C500/Ultralytics 环境支持情况使用）。断点续训：

`--batch` 是全局 batch size，会由 Ultralytics 在 8 张卡间分配；如需每卡约 16 张图片，可设置 `--batch 128`（实际可用值取决于显存）。

训练过程指标和累计耗时保存在 `runs/<name>/results.csv`（`time` 列，单位为秒）；建议同时将终端输出保存：`bash run_train.sh ... 2>&1 | tee logs/train_yolo26n.log`。

```bash
bash run_train.sh --model runs/yolo26n_coco/weights/last.pt --resume
```

完整参数可运行 `python yolo/train_yolo.py --help` 查看。常用参数还包括 `--workers`、`--cache`、`--seed`、`--project`。

## 验证训练结果

在容器内运行以下命令（8 卡验证）：

```bash
bash run_validate.sh --model runs/yolo26n_coco/weights/best.pt \
  --data /mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml \
  --device 0,1,2,3,4,5,6,7 2>&1 | tee logs/validate_yolo26n_best.log
```

验证预训练示例权重：

```bash
bash run_validate.sh --model yolo26n.pt \
  --data /mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml \
  --device 0,1,2,3,4,5,6,7 2>&1 | tee logs/validate_yolo26n_gold.log
```

该脚本输出 mAP50-95、mAP50、mAP75 和各类别 AP；推理流程可继续参考上级目录的 `eval_yolo_mx.py`。

## 环境检查

```bash
python -c 'import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())'
```

请在 C500 的 maca-pytorch/Ultralytics 环境中运行；若数据或权重路径不同，使用命令行参数覆盖默认值。
