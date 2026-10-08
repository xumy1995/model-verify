# CUDA YOLO26 训练

本目录使用与 `mx-c500_train/yolo/train_yolo.py` 一致的训练超参数，在 8 张 NVIDIA A100 上通过 CUDA 从官方起始权重训练 YOLO26。需使用与 C500 相同的 Ultralytics 8.4.115；PyTorch/CUDA 运行时因硬件不同而不同。

在已有 `cuda_train/venv-cuda-py312` 中安装依赖：

```bash
uv pip install --python ../venv-cuda-py312/bin/python 'ultralytics==8.4.115'
```

## 环境检查

```bash
cd /data/xumengying/model-verify/cuda_train/yolo
../venv-cuda-py312/bin/python -c 'import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

## 开始训练

```bash
cd /data/xumengying/model-verify/cuda_train/yolo
bash run_train.sh 2>&1 | tee logs/train_yolo26n_official.log
```

默认配置使用：

- 数据：`/data/xumengying/models_and_datasets/coco_yolo_format/coco.yaml`
- 起始权重：`yolo26n-objv1-150.pt`（Ultralytics 自动从本地缓存或官方来源获取）
- 设备：`0,1,2,3,4,5,6,7`
- 输出：`cuda_train/yolo/runs/yolo26n_official_recipe/`

完整训练参数与 MX-C500 版本保持一致，具体参数可通过以下命令查看：

```bash
../venv-cuda-py312/bin/python train_yolo.py --help
```

断点续训示例：

```bash
bash run_train.sh --model yolo/runs/yolo26n_official_recipe/weights/last.pt --resume
```
