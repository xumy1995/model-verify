"""Evaluate a trained torchvision ResNet-50 checkpoint on Kaggle ILSVRC."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


class ILSVRCValidation(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        lines = (root / "ImageSets/CLS-LOC/val.txt").read_text().splitlines()
        classes = sorted(
            p.name for p in (root / "Data/CLS-LOC/train").iterdir() if p.is_dir()
        )
        class_to_idx = {name: i for i, name in enumerate(classes)}
        self.samples = []
        for line in lines:
            name = line.split()[0]
            annotation = ET.parse(
                root / "Annotations/CLS-LOC/val" / f"{name}.xml"
            ).getroot()
            synset = annotation.find("./object/name").text
            self.samples.append(
                (root / "Data/CLS-LOC/val" / f"{name}.JPEG", class_to_idx[synset])
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ResNet-50 accuracy and performance on ImageNet-1K"
    )
    p.add_argument("--checkpoint", default="checkpoints/resnet50_v2/last.pth")
    p.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/ILSVRC")
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model = models.resnet50(weights=None)
    model.load_state_dict(state)
    return model


def run(args, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print(
            json.dumps(
                {"event": "skip", "device": "cuda", "reason": "CUDA unavailable"}
            )
        )
        return
    device = torch.device(device_name)
    root = Path(args.data)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_transform = transforms.Compose(
        [
            transforms.Resize(232, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ]
    )
    train_dir = root / "Data/CLS-LOC/train"
    if train_dir.is_dir() and (root / "ImageSets/CLS-LOC/val.txt").is_file():
        dataset = ILSVRCValidation(root, val_transform)
    else:
        dataset = ImageFolder(root / "val", val_transform)
    if args.max_samples:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_samples, len(dataset)))
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    t0 = time.perf_counter()
    model = load_model(args.checkpoint).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model.half()
    sync(device)
    load_time = time.perf_counter() - t0
    total = measured = top1 = top5 = 0
    preprocess_time = inference_time = measured_wall_time = 0.0
    print("\n========================")
    print(f"Running on: {device}")
    print("========================")
    with torch.inference_mode():
        progress = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"{device} ImageNet eval",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, (images, labels) in progress:
            batch_start = time.perf_counter()
            prep_start = time.perf_counter()
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            if args.fp16 and device.type == "cuda":
                images = images.half()
            sync(device)
            prep_elapsed = time.perf_counter() - prep_start
            infer_start = time.perf_counter()
            with torch.autocast(
                device_type="cuda", enabled=args.fp16 and device.type == "cuda"
            ):
                logits = model(images)
            sync(device)
            infer_elapsed = time.perf_counter() - infer_start
            if batch_index >= args.warmup_batches:
                measured += labels.numel()
                measured_wall_time += time.perf_counter() - batch_start
                preprocess_time += prep_elapsed
                inference_time += infer_elapsed
            pred = logits.topk(5, dim=1).indices
            top1 += (pred[:, 0] == labels).sum().item()
            top5 += (pred == labels[:, None]).any(dim=1).sum().item()
            total += labels.numel()
            progress.set_postfix(
                samples=total,
                top1=f"{100.0 * top1 / total:.2f}%",
            )
    throughput = measured / measured_wall_time if measured_wall_time else 0.0
    inference_throughput = measured / inference_time if inference_time else 0.0
    latency = 1000.0 * inference_time / measured if measured else 0.0
    print(f"Model path       : {args.checkpoint}")
    print(f"Dataset path     : {args.data}")
    print("Split            : validation")
    print(f"Samples          : {total}")
    print(f"Labeled samples  : {total}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Warmup batches   : {args.warmup_batches}")
    print(
        f"Precision        : {'fp16' if args.fp16 and device.type == 'cuda' else 'fp32'}"
    )
    print(f"Model load time  : {load_time:.3f}s")
    print(f"Preprocess time  : {preprocess_time:.3f}s")
    print(f"Inference time   : {inference_time:.3f}s")
    print(f"Measured wall    : {measured_wall_time:.3f}s")
    print(f"Throughput       : {throughput:.2f} samples/s")
    print(f"Infer throughput : {inference_throughput:.2f} samples/s")
    print(f"Avg infer latency: {latency:.3f} ms/sample")
    print(f"Top-1 accuracy   : {100.0 * top1 / total:.2f}%")
    print(f"Top-5 accuracy   : {100.0 * top5 / total:.2f}%")


def main():
    args = parse_args()
    for device in args.devices:
        run(args, device)


if __name__ == "__main__":
    main()
