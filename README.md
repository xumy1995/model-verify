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
| [YOLO26](https://huggingface.co/Ultralytics/YOLO26) | [COCO(原始格式)](https://docs.ultralytics.com/datasets/detect/coco) |
| [VIT-B/16](https://huggingface.co/google/vit-base-patch16-224) | [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| [clip-vit-b/32](https://huggingface.co/openai/clip-vit-base-patch32)| [ImageNet-1k(zero shot)](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |


# 评测脚本

## ResNet-50
```bash
python eval_resnet.py --devices cuda | tee logs/resnet50-imagenet-1k-val-cuda.log
```

## Detr-Resnet-50
```bash
python eval_detr.py --devices cuda | tee logs/detr-resnet-50-coco-val-cuda.log
```

## YOLO26
```bash
uv pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
python eval_yolo.py --model /data/xumengying/models_and_datasets/YOLO26/yolo26n.pt | tee logs/yolo26n-coco-val-cuda.log
python eval_yolo.py --model /data/xumengying/models_and_datasets/YOLO26/yolo26s.pt | tee logs/yolo26s-coco-val-cuda.log
python eval_yolo.py --model /data/xumengying/models_and_datasets/YOLO26/yolo26m.pt | tee logs/yolo26m-coco-val-cuda.log
python eval_yolo.py --model /data/xumengying/models_and_datasets/YOLO26/yolo26l.pt | tee logs/yolo26l-coco-val-cuda.log
python eval_yolo.py --model /data/xumengying/models_and_datasets/YOLO26/yolo26x.pt | tee logs/yolo26x-coco-val-cuda.log
```

## VIT-B/16
```bash
python eval_vit.py --devices cuda | tee logs/VIT-B-16-imagenet-1k-val-cuda.log
```

## clip-vit-b/32
