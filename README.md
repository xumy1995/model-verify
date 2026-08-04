# 环境配置
```bash
uv venv ./venv-cuda-py312 --python 3.12
source ./venv-cuda-py312/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers datasets Pillow pycocotools timm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 评测模型和数据集

| Model | Dataset |
| ----  | --------|
| [ResNet-50](https://huggingface.co/microsoft/resnet-50) | [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| [detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) | [COCO](https://huggingface.co/datasets/detection-datasets/coco) |


# 评测脚本

```bash
python eval_imagenet.py --devices cuda | tee logs/resnet50-imagenet-1k-val-cuda.log
python eval_coco.py --devices cuda | tee logs/detr-resnet-50-coco-val-cuda.log
```
