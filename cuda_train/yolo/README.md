# CUDA YOLO26 训练

本目录使用与 `mx-c500_train/yolo/train_yolo.py` 一致的训练超参数，在 8 张 NVIDIA A100 上通过 CUDA 从官方起始权重训练 YOLO26。需使用与 C500 相同的 Ultralytics 8.4.115；PyTorch/CUDA 运行时因硬件不同而不同。

在已有 `cuda_train/venv-cuda-py312` 中安装依赖：

```bash
cd /data/xumengying/model-verify/cuda_train/yolo
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
mkdir -p logs
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

## 训练结果与验证

245 轮训练于 2026-10-08 完成（训练计时 41,592 秒）；最佳轮次为 245。
完整模型保存在 `runs/yolo26n_official_recipe/weights/best.pt`，逐轮指标备份在
[`logs/train_yolo26n_official_results.csv`](logs/train_yolo26n_official_results.csv)。首次启动因缺失数据标签中断，仅生成
`args.yaml`；该无效目录已删除，完整训练结果由 Ultralytics 自动生成的 `-2` 目录
归位到标准目录。`runs/yolo26n_official_recipe/args.yaml` 中仍保留运行时原始的
`name: yolo26n_official_recipe-2`，便于追溯。原始终端日志
`logs/train_yolo26n_official.log` 记录了完整训练（末尾另有一次
`run_train.sh` 命令错误，不影响已完成的训练）。

从本目录运行单卡完整 COCO val 独立评测（batch 16、输入 640；与 C500 验证脚本一致）：

```bash
mkdir -p logs
bash run_validate.sh > logs/validate_yolo26n_best.log 2>&1
```

验证结果写入 `runs/validation_best/`。评测日志：

- CUDA 完整日志：[`logs/validate_yolo26n_best.log`](logs/validate_yolo26n_best.log)
- CUDA 指标摘要：[`logs/validate_yolo26n_best_summary.txt`](logs/validate_yolo26n_best_summary.txt)
- MX-C500 对照日志：[`mx-c500_train/yolo/logs/validate_yolo26n_official_best.log`](../../mx-c500_train/yolo/logs/validate_yolo26n_official_best.log)

| 模型 | mAP50-95 | mAP50 | mAP75 |
|---|---:|---:|---:|
| CUDA YOLO26n best.pt | 0.3765 | 0.5354 | 0.4135 |
| MX-C500 YOLO26n best.pt | 0.3821 | 0.5337 | 0.4133 |
