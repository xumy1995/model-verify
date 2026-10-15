"""Evaluate a locally trained OpenCLIP checkpoint on ImageNet-1K validation."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import open_clip
import torch
from torch.utils.data import DataLoader

from cuda_train.vit.train_vit_b16 import ILSVRCValidation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="cuda_train/clip/logs/cc3m_vit_b32_8xa100/checkpoints/epoch_32.pt")
    parser.add_argument("--data", default="/data/xumengying/models_and_datasets/ILSVRC")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="cuda_train/clip/results/imagenet_epoch32.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started = time.perf_counter()
    root = Path(args.data)
    classes = sorted(path.name for path in (root / "Data/CLS-LOC/train").iterdir() if path.is_dir())
    mapping = root.parent / "LOC_synset_mapping.txt"
    expected_classes = [line.split()[0] for line in mapping.read_text().splitlines() if line.strip()]
    if classes != expected_classes or len(classes) != 1000:
        raise ValueError("ImageNet synsets do not match the standard 1000-class ordering")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, precision="fp32")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    epoch = checkpoint["epoch"]
    del checkpoint, state
    model = model.to("cuda").eval()
    dataset = ILSVRCValidation(root, preprocess, classes)
    if len(dataset) != 50000 or len({path for path, _ in dataset.samples}) != 50000:
        raise ValueError("Expected 50,000 distinct ImageNet validation images")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    print(f"Checkpoint: {args.checkpoint}; epoch: {epoch}; samples: {len(dataset)}", flush=True)
    print(f"Preprocess: {preprocess}", flush=True)
    with torch.inference_mode():
        classifier = open_clip.build_zero_shot_classifier(
            model,
            tokenizer=open_clip.get_tokenizer("ViT-B-32"),
            classnames=open_clip.IMAGENET_CLASSNAMES,
            templates=open_clip.OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device="cuda",
            use_tqdm=True,
        )
        total = top1 = top5 = 0
        for images, labels in loader:
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            features = model.encode_image(images, normalize=True)
            predictions = (features @ classifier).topk(5, dim=1).indices
            top1 += (predictions[:, 0] == labels).sum().item()
            top5 += (predictions == labels[:, None]).any(dim=1).sum().item()
            total += labels.numel()
            if total % (args.batch_size * 20) == 0 or total == len(dataset):
                print(f"Samples: {total}/{len(dataset)}; Top-1: {100 * top1 / total:.3f}%; Top-5: {100 * top5 / total:.3f}%", flush=True)
    with open(args.checkpoint, "rb") as checkpoint_file:
        checksum = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checksum,
        "epoch": epoch,
        "model": "ViT-B-32",
        "data": str(root),
        "samples": total,
        "classes": len(classes),
        "protocol": "ImageNet-1K zero-shot, OpenCLIP IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES",
        "templates": len(open_clip.OPENAI_IMAGENET_TEMPLATES),
        "precision": "fp32",
        "preprocess": str(preprocess),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "correct_top1": top1,
        "correct_top5": top5,
        "top1_percent": 100 * top1 / total,
        "top5_percent": 100 * top5 / total,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {package: version(package) for package in ("torch", "torchvision", "open_clip_torch", "webdataset", "pillow")},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
