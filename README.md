# 模型评测结果汇总

结果来自 `cuda/logs/` 与 `mx-c500/logs/`，指标单位在“评测项”中注明。运行如下命令可自动生成本文表格：
```
python generate_summary.py > summary.md
```

## 精度对比

| 模型 | 数据集 | 评测项 | cuda指标值 | mx-c500指标值 | 对比结果 |
|---|---|---|---:|---:|---:|
| ResNet-50 | ImageNet-1k val | Top-1 accuracy (%) | 80.15 | 80.15 | +0.00 % |
| ResNet-50 | ImageNet-1k val | Top-5 accuracy (%) | 94.50 | 94.50 | +0.00 % |
| ViT-B/16 | ImageNet-1k val | Top-1 accuracy (%) | 80.31 | 80.31 | +0.00 % |
| ViT-B/16 | ImageNet-1k val | Top-5 accuracy (%) | 95.49 | 95.49 | +0.00 % |
| CLIP ViT-B/32 (zero-shot) | ImageNet-1k val | Top-1 accuracy (%) | 60.83 | 60.83 | +0.00 % |
| CLIP ViT-B/32 (zero-shot) | ImageNet-1k val | Top-5 accuracy (%) | 85.54 | 85.53 | -0.01 % |
| DETR-ResNet-50 | COCO val | AP (IoU 0.50:0.95) | 0.571 | 0.571 | +0.00 % |
| DETR-ResNet-50 | COCO val | AP50 | 0.614 | 0.615 | +0.00 % |
| DETR-ResNet-50 | COCO val | AP75 | 0.435 | 0.435 | +0.00 % |
| YOLO26n | COCO val | mAP50-95 | 0.3950 | 0.3949 | -0.00 % |
| YOLO26n | COCO val | mAP50 | 0.5502 | 0.5501 | -0.00 % |
| YOLO26n | COCO val | mAP75 | 0.4297 | 0.4296 | -0.00 % |
| YOLO26s | COCO val | mAP50-95 | 0.4717 | 0.4715 | -0.00 % |
| YOLO26s | COCO val | mAP50 | 0.6384 | 0.6380 | -0.00 % |
| YOLO26s | COCO val | mAP75 | 0.5157 | 0.5158 | +0.00 % |
| YOLO26m | COCO val | mAP50-95 | 0.5180 | 0.5180 | -0.00 % |
| YOLO26m | COCO val | mAP50 | 0.6907 | 0.6907 | +0.00 % |
| YOLO26m | COCO val | mAP75 | 0.5667 | 0.5667 | +0.00 % |
| YOLO26l | COCO val | mAP50-95 | 0.5375 | 0.5376 | +0.00 % |
| YOLO26l | COCO val | mAP50 | 0.7088 | 0.7089 | +0.00 % |
| YOLO26l | COCO val | mAP75 | 0.5886 | 0.5886 | -0.00 % |
| YOLO26x | COCO val | mAP50-95 | 0.5626 | 0.5626 | +0.00 % |
| YOLO26x | COCO val | mAP50 | 0.7352 | 0.7353 | +0.00 % |
| YOLO26x | COCO val | mAP75 | 0.6159 | 0.6161 | +0.00 % |

## 性能对比

单位: ms/sample

| 模型 | 数据集 | 评测项 | cuda指标值 | mx-c500指标值 | 对比结果 |
|---|---|---|---:|---:|---:|
| ResNet-50 | ImageNet-1k val | 推理耗时 | 0.3503 | 0.3552 | 98.63% |
| ResNet-50 | ImageNet-1k val | 端到端耗时 | 0.3814 | 0.3717 | 102.61% |
| ViT-B/16 | ImageNet-1k val | 推理耗时 | 0.5278 | 1.0554 | **50.01%** |
| ViT-B/16 | ImageNet-1k val | 端到端耗时 | 0.5617 | 1.0725 | **52.38%** |
| CLIP ViT-B/32 (zero-shot) | ImageNet-1k val | 推理耗时 | 0.2218 | 0.2921 | **75.93%** |
| CLIP ViT-B/32 (zero-shot) | ImageNet-1k val | 端到端耗时 | 0.2540 | 0.3089 | 82.23% |
| DETR-ResNet-50 | COCO val | 推理耗时 | 13.694 | 18.111 | **75.61%** |
| DETR-ResNet-50 | COCO val | 端到端耗时 | 14.802 | 18.783 | **78.80%** |
| YOLO26n | COCO val | 推理耗时 | 0.9823 | 0.9189 | 106.90% |
| YOLO26n | COCO val | 端到端耗时 | 1.4070 | 1.3309 | 105.72% |
| YOLO26s | COCO val | 推理耗时 | 1.3757 | 1.5145 | 90.84% |
| YOLO26s | COCO val | 端到端耗时 | 1.7983 | 1.9276 | 93.29% |
| YOLO26m | COCO val | 推理耗时 | 2.1111 | 2.6931 | **78.39%** |
| YOLO26m | COCO val | 端到端耗时 | 2.5288 | 3.1363 | 80.63% |
| YOLO26l | COCO val | 推理耗时 | 2.5991 | 3.4495 | **75.35%** |
| YOLO26l | COCO val | 端到端耗时 | 3.0187 | 3.9130 | **77.15%** |
| YOLO26x | COCO val | 推理耗时 | 4.2067 | 5.7145 | **73.61%** |
| YOLO26x | COCO val | 端到端耗时 | 4.6206 | 6.1489 | **75.15%** |

原始日志：[`cuda/logs/`](cuda/logs/)、[`mx-c500/logs/`](mx-c500/logs/)。

对比规则：精度为 MX-C500−CUDA 的百分点差；耗时性能为 CUDA耗时/MX-C500耗时。加粗表示精度掉点超过 5 个百分点或性能低于 CUDA 的 80%。
