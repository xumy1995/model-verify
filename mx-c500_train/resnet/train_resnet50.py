"""Train ResNet-50 on ImageNet with TorchVision reference-style recipes.

Default target: TorchVision ResNet50_Weights.IMAGENET1K_V2 / DEFAULT.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import InterpolationMode


BASELINE = {
    "v1": {
        "weights": "ResNet50_Weights.IMAGENET1K_V1",
        "acc1": 76.130,
        "acc5": 92.862,
        "recipe": "classic 90-epoch TorchVision reference recipe",
    },
    "v2": {
        "weights": "ResNet50_Weights.IMAGENET1K_V2 / DEFAULT",
        "acc1": 80.858,
        "acc5": 95.434,
        "recipe": "TorchVision new recipe",
    },
}


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 2:
            log_probs = F.log_softmax(logits, dim=1)
            return -(target * log_probs).sum(dim=1).mean()
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class ILSVRCValidation(torch.utils.data.Dataset):
    """Kaggle ILSVRC layout: flat val images plus ImageSets/CLS-LOC/val.txt."""

    def __init__(self, root: Path, transform, classes):
        self.root, self.transform = root, transform
        lines = (root / "ImageSets/CLS-LOC/val.txt").read_text().splitlines()
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.samples = []
        for line in lines:
            name = line.split()[0]
            xml_path = root / "Annotations/CLS-LOC/val" / f"{name}.xml"
            annotation = ET.parse(xml_path).getroot()
            synset = annotation.find("./object/name").text
            if synset not in self.class_to_idx:
                raise ValueError(
                    f"Validation synset {synset} is absent from training classes"
                )
            self.samples.append(
                (root / "Data/CLS-LOC/val" / f"{name}.JPEG", self.class_to_idx[synset])
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="/mnt/afs/xumengying/models_and_datasets/ILSVRC",
        help="ImageNet root",
    )
    p.add_argument("--recipe", choices=("v1", "v2"), default="v2")
    p.add_argument("--output-dir", default="checkpoints/resnet50_v2")
    p.add_argument("--log-file", default="logs/resnet50_v2_train.log")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--norm-weight-decay", type=float)
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--mixup-alpha", type=float)
    p.add_argument("--cutmix-alpha", type=float)
    p.add_argument("--train-crop-size", type=int)
    p.add_argument("--val-resize-size", type=int)
    p.add_argument("--val-crop-size", type=int, default=224)
    p.add_argument("--random-erase", type=float)
    p.add_argument("--scheduler", choices=("step", "cosine"), default=None)
    p.add_argument("--lr-step-size", type=int, default=30)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-batches", type=int, help="debug limit per epoch")
    p.add_argument("--val-batches", type=int, help="debug limit for validation")
    p.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="print progress every N train batches",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def apply_recipe_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.recipe == "v1":
        defaults = {
            "epochs": 90,
            "batch_size": 256,
            "lr": 0.1,
            "weight_decay": 1e-4,
            "norm_weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
            "train_crop_size": 224,
            "val_resize_size": 256,
            "random_erase": 0.0,
            "scheduler": "step",
        }
    else:
        defaults = {
            "epochs": 600,
            "batch_size": 128,
            "lr": 0.5,
            "weight_decay": 2e-5,
            "norm_weight_decay": 0.0,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "cutmix_alpha": 1.0,
            "train_crop_size": 176,
            "val_resize_size": 232,
            "random_erase": 0.1,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "amp": True,
        }
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, False):
            setattr(args, key, value)
    if args.smoke:
        args.epochs = 1
        args.train_batches = 1
        args.val_batches = 1
        args.workers = min(args.workers, 2)
    return args


def build_transforms(args: argparse.Namespace):
    interpolation = (
        InterpolationMode.BICUBIC if args.recipe == "v2" else InterpolationMode.BILINEAR
    )
    train_ops = [
        transforms.RandomResizedCrop(args.train_crop_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
    ]
    if args.recipe == "v2":
        train_ops.append(transforms.TrivialAugmentWide(interpolation=interpolation))
    train_ops += [
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if args.random_erase:
        train_ops.append(transforms.RandomErasing(p=args.random_erase))
    val_tf = transforms.Compose(
        [
            transforms.Resize(args.val_resize_size, interpolation=interpolation),
            transforms.CenterCrop(args.val_crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(train_ops), val_tf


def add_weight_decay(model: nn.Module, weight_decay: float, norm_weight_decay: float):
    norm_classes = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.LocalResponseNorm,
    )
    params = []
    memo = set()
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or parameter in memo:
                continue
            memo.add(parameter)
            wd = (
                norm_weight_decay
                if isinstance(module, norm_classes) or name.endswith("bias")
                else weight_decay
            )
            params.append({"params": [parameter], "weight_decay": wd})
    return params


def one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    out = torch.full((labels.size(0), num_classes), off, device=labels.device)
    out.scatter_(1, labels[:, None], on)
    return out


def mix_batch(images: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace):
    alpha, mode = 0.0, None
    if args.mixup_alpha and args.cutmix_alpha:
        alpha, mode = (
            (args.mixup_alpha, "mixup")
            if torch.rand(()) < 0.5
            else (args.cutmix_alpha, "cutmix")
        )
    elif args.mixup_alpha:
        alpha, mode = args.mixup_alpha, "mixup"
    elif args.cutmix_alpha:
        alpha, mode = args.cutmix_alpha, "cutmix"
    if not alpha:
        return images, labels

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    perm = torch.randperm(images.size(0), device=images.device)
    if mode == "cutmix":
        _, _, h, w = images.shape
        cut_ratio = math.sqrt(1 - lam)
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
        cx, cy = (
            torch.randint(w, (1,), device=images.device).item(),
            torch.randint(h, (1,), device=images.device).item(),
        )
        x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        x2, y2 = min(cx + cut_w // 2, w), min(cy + cut_h // 2, h)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    else:
        images = lam * images + (1 - lam) * images[perm]
    targets = lam * one_hot(labels, 1000, args.label_smoothing) + (1 - lam) * one_hot(
        labels[perm], 1000, args.label_smoothing
    )
    return images, targets


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    _, pred = output.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum(0).mul_(100.0 / target.numel()).item()
        for k in topk
    ]


def emit(record: dict, log_file: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(line + "\n")


def distributed_info():
    enabled = dist.is_available() and dist.is_initialized()
    return (
        enabled,
        (dist.get_rank() if enabled else 0),
        (dist.get_world_size() if enabled else 1),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
):
    model.eval()
    total = top1 = top5 = loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, labels)
        a1, a5 = accuracy(output, labels)
        bs = labels.size(0)
        total += bs
        top1 += a1 * bs
        top5 += a5 * bs
        loss_sum += loss.item() * bs
        if max_batches and i + 1 >= max_batches:
            break
    enabled, _, world = distributed_info()
    values = torch.tensor(
        [loss_sum, top1, top5, total], device=device, dtype=torch.float64
    )
    if enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    loss_sum, top1, top5, total = values.tolist()
    return {
        "loss": loss_sum / total,
        "acc1": top1 / total,
        "acc5": top5 / total,
        "samples": int(total),
        "world_size": world,
    }


def main() -> None:
    args = apply_recipe_defaults(parse_args())
    ddp = (
        "RANK" in __import__("os").environ and "WORLD_SIZE" in __import__("os").environ
    )
    if ddp:
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    enabled, rank, world = distributed_info()
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    device = torch.device(
        f"cuda:{local_rank}"
        if enabled
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = build_transforms(args)
    data_root = Path(args.data)
    kaggle_root = data_root / "Data/CLS-LOC"
    if (kaggle_root / "train").is_dir() and (
        data_root / "ImageSets/CLS-LOC/val.txt"
    ).is_file():
        train_ds = ImageFolder(kaggle_root / "train", train_tf)
        val_ds = ILSVRCValidation(data_root, val_tf, train_ds.classes)
    else:
        train_ds = ImageFolder(data_root / "train", train_tf)
        val_ds = ImageFolder(data_root / "val", val_tf)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if enabled else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if enabled else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = models.resnet50(weights=None).to(device)
    if enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    optimizer = torch.optim.SGD(
        add_weight_decay(model, args.weight_decay, args.norm_weight_decay),
        lr=args.lr,
        momentum=args.momentum,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if enabled else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
    else:
        start_epoch = 0

    config = vars(args) | {
        "device": str(device),
        "world_size": world,
        "baseline": BASELINE[args.recipe],
    }
    if rank == 0:
        emit({"event": "config", **config}, args.log_file)

    if args.eval_only:
        metrics = evaluate(model, val_loader, device, args.val_batches)
        if rank == 0:
            emit({"event": "eval", **metrics}, args.log_file)
        return

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        seen = 0
        running = 0.0
        if args.warmup_epochs and epoch < args.warmup_epochs:
            warmup_factor = float(epoch + 1) / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup_factor
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, train_targets = mix_batch(images, labels, args)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = criterion(model(images), train_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = labels.size(0)
            seen += bs
            running += loss.item() * bs
            if (i + 1) % args.log_interval == 0:
                progress = {
                    "event": "batch",
                    "epoch": epoch + 1,
                    "batch": i + 1,
                    "train_loss": running / seen,
                    "samples": seen,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if rank == 0:
                    emit(progress, args.log_file)
            if args.train_batches and i + 1 >= args.train_batches:
                break
        if not (args.warmup_epochs and epoch < args.warmup_epochs):
            scheduler.step()
        metrics = evaluate(model, val_loader, device, args.val_batches)
        row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train_loss": running / seen,
            "train_samples": seen,
            "seconds": round(time.time() - t0, 3),
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        if rank == 0:
            emit(row, args.log_file)
            state = model.module.state_dict() if enabled else model.state_dict()
            torch.save(
                {
                    "model": state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "args": vars(args),
                },
                Path(args.output_dir) / "last.pth",
            )
        if args.smoke:
            break
    if enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
