# 获取镜像

## 1. 拉取沐曦开发镜像 

- 名称版本：`maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64`
- 镜像下载地址：https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_kind=AI&dimension=docker&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=pytorch&frame_version=2.10&python_version=3.12
- 需要注册登录，点击【docker pull命令复制】

## 2. 加载镜像

```bash
sudo docker run -it \
  --name maca-pytorch-test \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --group-add video \
  --shm-size=16g \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64   /bin/bash
```

## 3. 进入镜像并验证pytorch可用
```bash
sudo docker exec -it maca-pytorch-test /bin/bash
```

`exit`退出后，再次进入镜像：
```bash
sudo docker start maca-pytorch-test && sudo docker exec -it maca-pytorch-test /bin/bash
```

验证python可用
```
root@2599eb74c47a:/# python
Python 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:45:31) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name()
'MetaX C500'
```

# 挂载模型、数据和代码目录

宿主机上的模型和 ImageNet 数据位于 `/mnt/afs/xumengying/models_and_datasets`，
评测代码位于本仓库。启动容器时将这些目录挂载到容器内相同路径（容器已创建时，
挂载不能通过 `docker exec` 补加，需要重新 `docker run`）：

```bash
sudo docker rm -f maca-pytorch-test 2>/dev/null || true
sudo docker run -it --name maca-pytorch-test \
  --device=/dev/dri --device=/dev/mxcd --group-add video \
  --shm-size=16g \
  -v /mnt/afs/xumengying/model-verify:/workspace/model-verify \
  -v /mnt/afs/xumengying/models_and_datasets:/mnt/afs/xumengying/models_and_datasets \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64 \
  /bin/bash
```

若宿主机路径在容器中已经可见，可省略第二个 `-v`；进入容器后用
`ls /mnt/afs/xumengying/models_and_datasets/resnet-50` 和
`ls /mnt/afs/xumengying/models_and_datasets/imagenet-1k` 确认挂载成功。

# 评测

进入`mx-c500`并创建`logs`目录
```bash
cd /workspace/model-verify/mx-c500
mkdir -p logs
```

## Resnet-50

```bash
python -m pip install transformers datasets Pillow tqdm
python eval_resnet_mx.py \
  --model-path /mnt/afs/xumengying/models_and_datasets/resnet-50 \
  --dataset-path /mnt/afs/xumengying/models_and_datasets/imagenet-1k \
  --batch-size 64 --num-workers 4 \
  | tee logs/resnet50-imagenet-1k-val-mx-c500.log
```


## DETR-ResNet-50
```bash
python -m pip install pycocotools timm
python eval_detr_mx.py \
  --model-path /mnt/afs/xumengying/models_and_datasets/detr-resnet-50 \
  --dataset-path /mnt/afs/xumengying/models_and_datasets/coco \
  --split val --batch-size 4 --num-workers 4 \
  | tee logs/detr-resnet-50-coco-val-mx-c500.log
```

## YOLO26

```bash
python -m pip install ultralytics
# ultralytics 会导入 OpenCV；基础镜像若缺少 libGL.so.1，先安装运行库
apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0
python eval_yolo_mx.py --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26n.pt | tee logs/yolo26n-coco-val-mx-c500.log
python eval_yolo_mx.py --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26s.pt | tee logs/yolo26s-coco-val-mx-c500.log
python eval_yolo_mx.py --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26m.pt | tee logs/yolo26m-coco-val-mx-c500.log
python eval_yolo_mx.py --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26l.pt | tee logs/yolo26l-coco-val-mx-c500.log
python eval_yolo_mx.py --model /mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26x.pt | tee logs/yolo26x-coco-val-mx-c500.log
```

## VIT-B/16
```bash
python eval_vit_mx.py \
  --model-path /mnt/afs/xumengying/models_and_datasets/vit-base-patch16-224 \
  --dataset-path /mnt/afs/xumengying/models_and_datasets/imagenet-1k \
  --batch-size 64 --num-workers 4 \
  --devices cuda \
  | tee logs/VIT-B-16-imagenet-1k-val-mx-c500.log

算子级性能分析（不需要 ImageNet 数据，默认固定 batch=64 输入；结果按设备耗时排序）：
```bash
python profile_vit_ops.py --model-path /mnt/afs/xumengying/models_and_datasets/vit-base-patch16-224 \
  --batch-size 64 --warmup 5 --steps 10 --trace logs/vit-profile-mx.json
```
脚本固定单卡 `cuda:0`（可用 `--device cuda:1` 修改），并额外打印 embeddings、12 个
Transformer layer、layernorm、classifier 的平均阶段耗时；总和可与评测中的 inference time 对照。
重点关注 `aten::matmul`/`aten::bmm`、`aten::linear` 及其对应的设备 kernel；将生成的 JSON 用 Chrome `chrome://tracing` 或 Perfetto 打开。
```

## clip-vit-b/32 + Imagenet-1k
```bash
python eval_clip_mx.py \
  --model-path /mnt/afs/xumengying/models_and_datasets/clip-vit-base-patch32 \
  --dataset-path /mnt/afs/xumengying/models_and_datasets/imagenet-1k \
  --batch-size 64 --num-workers 4 \
  --devices cuda \
  | tee logs/clip-vit-b-32-imagenet-1k-val-mx-c500.log
```
