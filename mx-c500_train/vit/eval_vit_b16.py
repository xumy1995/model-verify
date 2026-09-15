#!/usr/bin/env python3
"""Evaluate a trained TorchVision ViT-B/16 checkpoint on ImageNet-1K."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from train_vit_b16 import ILSVRCValidation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="vit/checkpoints/vit_b16_v1_ddp/best.pth")
    parser.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/ILSVRC")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(path: str, weights: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_ema" if weights == "ema" else "model"]
    if weights == "ema":
        state = {key.removeprefix("module."): value for key, value in state.items() if key != "n_averaged"}
    model = models.vit_b_16(weights=None, num_classes=1000)
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MX-C500 CUDA-compatible device is unavailable")
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    root = Path(args.data)
    classes = sorted(path.name for path in (root / "Data/CLS-LOC/train").iterdir() if path.is_dir())
    dataset = ILSVRCValidation(root, transform, classes)
    if args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    started = time.perf_counter()
    model = load_model(args.checkpoint, args.weights).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model.half()
    sync(device)
    load_seconds = time.perf_counter() - started
    total = measured = top1 = top5 = 0
    wall_seconds = inference_seconds = transfer_seconds = 0.0
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            batch_started = time.perf_counter()
            transfer_started = time.perf_counter()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.fp16 and device.type == "cuda":
                images = images.half()
            sync(device)
            transfer = time.perf_counter() - transfer_started
            inference_started = time.perf_counter()
            with torch.autocast("cuda", enabled=args.fp16 and device.type == "cuda"):
                logits = model(images)
            sync(device)
            inference = time.perf_counter() - inference_started
            predictions = logits.topk(5, dim=1).indices
            top1 += (predictions[:, 0] == labels).sum().item()
            top5 += (predictions == labels[:, None]).any(dim=1).sum().item()
            total += labels.numel()
            if index >= args.warmup_batches:
                measured += labels.numel()
                transfer_seconds += transfer
                inference_seconds += inference
                wall_seconds += time.perf_counter() - batch_started
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint weights: {args.weights}")
    print(f"Device           : {device}")
    print(f"Precision        : {'fp16' if args.fp16 and device.type == 'cuda' else 'fp32'}")
    print(f"Samples          : {total}")
    print(f"Model load time  : {load_seconds:.3f}s")
    print(f"Transfer time    : {transfer_seconds:.3f}s")
    print(f"Inference time   : {inference_seconds:.3f}s")
    print(f"Throughput       : {measured / wall_seconds if wall_seconds else 0:.2f} samples/s")
    print(f"Infer throughput : {measured / inference_seconds if inference_seconds else 0:.2f} samples/s")
    print(f"Avg infer latency: {1000 * inference_seconds / measured if measured else 0:.3f} ms/sample")
    print(f"Top-1 accuracy   : {100 * top1 / total:.3f}%")
    print(f"Top-5 accuracy   : {100 * top5 / total:.3f}%")


if __name__ == "__main__":
    main()
