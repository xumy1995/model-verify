#!/usr/bin/env python3
"""Train TorchVision ViT-B/16 on ImageNet using its official V1 recipe."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


OFFICIAL_BASELINE = {
    "weights": "ViT_B_16_Weights.IMAGENET1K_V1",
    "recipe": "TorchVision modified DeiT recipe",
    "acc1": 81.072,
    "acc5": 95.318,
}


class ILSVRCValidation(Dataset):
    """Kaggle ILSVRC validation split: flat images with XML labels."""

    def __init__(self, root: Path, transform, classes: list[str]) -> None:
        self.transform = transform
        class_to_idx = {name: i for i, name in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        lines = (root / "ImageSets/CLS-LOC/val.txt").read_text().splitlines()
        for line in lines:
            image_id = line.split()[0]
            annotation = ET.parse(
                root / "Annotations/CLS-LOC/val" / f"{image_id}.xml"
            ).getroot()
            synset = annotation.findtext("./object/name")
            if synset not in class_to_idx:
                raise ValueError(f"Unknown validation synset: {synset}")
            self.samples.append(
                (root / "Data/CLS-LOC/val" / f"{image_id}.JPEG", class_to_idx[synset])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


class RASampler(Sampler[int]):
    """TorchVision/DeiT repeated-augmentation distributed sampler."""

    def __init__(self, dataset: Dataset, repetitions: int = 3, seed: int = 0) -> None:
        self.dataset = dataset
        self.repetitions = repetitions
        self.seed = seed
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) * repetitions / self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.num_selected_samples = len(dataset) // 256 * 256 // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        indices = [index for index in indices for _ in range(self.repetitions)]
        indices += indices[: self.total_size - len(indices)]
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices[: self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__()
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long))
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, reset: bool = False) -> None:
        source = model.state_dict()
        decay = 0.0 if reset or self.n_averaged.item() == 0 else self.decay
        for name, value in self.module.state_dict().items():
            value.copy_(value * decay + source[name].detach() * (1.0 - decay))
        self.n_averaged.add_(1)
        if reset:
            self.n_averaged.zero_()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/ILSVRC")
    parser.add_argument("--output-dir", default="vit/checkpoints/vit_b16_v1_ddp")
    parser.add_argument("--log-file", default="vit/logs/vit_b16_v1_ddp.jsonl")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512, help="Per-device batch size")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--warmup-start-factor", type=float, default=0.033)
    parser.add_argument("--label-smoothing", type=float, default=0.11)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--ra-reps", type=int, default=3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99998)
    parser.add_argument("--ema-steps", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--train-batches", type=int)
    parser.add_argument("--val-batches", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=65536.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def distributed_info() -> tuple[bool, int, int, int]:
    enabled = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def emit(record: dict, log_file: str, rank: int) -> None:
    if rank != 0:
        return
    line = json.dumps(
        {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record},
        sort_keys=True,
    )
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def build_datasets(data_root: Path):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(interpolation=InterpolationMode.BILINEAR, magnitude=9),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ]
    )
    kaggle = data_root / "Data/CLS-LOC"
    if (kaggle / "train").is_dir() and (data_root / "ImageSets/CLS-LOC/val.txt").is_file():
        train_dataset = ImageFolder(kaggle / "train", train_transform)
        val_dataset = ILSVRCValidation(data_root, val_transform, train_dataset.classes)
    else:
        train_dataset = ImageFolder(data_root / "train", train_transform)
        val_dataset = ImageFolder(data_root / "val", val_transform)
    if len(train_dataset.classes) != 1000:
        raise ValueError(f"Expected 1000 classes, found {len(train_dataset.classes)}")
    return train_dataset, val_dataset


def mixup_cutmix(images: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace):
    use_cutmix = random.random() < 0.5
    alpha = args.cutmix_alpha if use_cutmix else args.mixup_alpha
    targets = F.one_hot(labels, num_classes=1000).to(images.dtype)
    if alpha <= 0:
        return images, targets
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    rolled_images, rolled_targets = images.roll(1, 0), targets.roll(1, 0)
    if use_cutmix:
        height, width = images.shape[-2:]
        ratio = 0.5 * math.sqrt(1.0 - lam)
        cx, cy = random.randrange(width), random.randrange(height)
        half_w, half_h = int(ratio * width), int(ratio * height)
        x1, x2 = max(cx - half_w, 0), min(cx + half_w, width)
        y1, y2 = max(cy - half_h, 0), min(cy + half_h, height)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = rolled_images[:, :, y1:y2, x1:x2]
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (width * height)
    else:
        images = images * lam + rolled_images * (1.0 - lam)
    return images, targets * lam + rolled_targets * (1.0 - lam)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, smoothing: float):
    targets = targets * (1.0 - smoothing) + smoothing / targets.shape[1]
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


@torch.inference_mode()
def evaluate(model, loader, device, max_batches: int | None):
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    criterion = nn.CrossEntropyLoss()
    for index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        batch = labels.numel()
        predictions = logits.topk(5, dim=1).indices
        sums += torch.tensor(
            [
                criterion(logits, labels).item() * batch,
                (predictions[:, 0] == labels).sum().item(),
                (predictions == labels[:, None]).any(dim=1).sum().item(),
                batch,
            ],
            dtype=torch.float64,
            device=device,
        )
        if max_batches and index + 1 >= max_batches:
            break
    if dist.is_initialized():
        dist.all_reduce(sums)
    loss, top1, top5, total = sums.tolist()
    return {"loss": loss / total, "acc1": 100 * top1 / total, "acc5": 100 * top5 / total, "samples": int(total)}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs, args.train_batches, args.val_batches = 1, 1, 1
        args.batch_size, args.workers = min(args.batch_size, 2), min(args.workers, 2)
        args.ema_steps = 1
        args.amp_init_scale = min(args.amp_init_scale, 1024.0)
    distributed, rank, world_size, local_rank = distributed_info()
    if not torch.cuda.is_available():
        raise RuntimeError("MX-C500 CUDA-compatible device is unavailable")
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(Path(args.data))
    if distributed:
        train_sampler = RASampler(train_dataset, args.ra_reps, args.seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers > 0 else None,
    )

    model = models.vit_b_16(weights=None, num_classes=1000).to(device)
    model_ema_decay = 1.0 - min(
        1.0,
        (1.0 - args.ema_decay) * world_size * args.batch_size * args.ema_steps / args.epochs,
    )
    ema = ModelEMA(model, model_ema_decay).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=args.warmup_start_factor, total_iters=args.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp, init_scale=args.amp_init_scale)
    start_epoch, best_acc1 = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint.get("best_acc1", 0.0)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if distributed else model

    emit({"event": "config", **vars(args), "world_size": world_size, "ema_decay_adjusted": model_ema_decay, "baseline": OFFICIAL_BASELINE}, args.log_file, rank)
    if args.eval_only:
        emit({"event": "eval", "weights": "ema", **evaluate(ema.module, val_loader, device, args.val_batches)}, args.log_file, rank)
        if distributed:
            dist.destroy_process_group()
        return

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = samples = 0
        started = time.time()
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, targets = mixup_cutmix(images, labels, args)
            sync_step = (batch_index + 1) % args.accumulation_steps == 0
            context = model.no_sync() if distributed and not sync_step else torch.enable_grad()
            with context:
                with torch.autocast("cuda", enabled=not args.no_amp):
                    logits = model(images)
                    loss = soft_cross_entropy(logits, targets, args.label_smoothing)
                    scaled_loss = loss / args.accumulation_steps
                scaler.scale(scaled_loss).backward()
            if sync_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if (batch_index // args.accumulation_steps) % args.ema_steps == 0:
                    ema.update(raw_model, reset=epoch < args.warmup_epochs)
            batch = labels.numel()
            loss_sum += loss.item() * batch
            samples += batch
            if (batch_index + 1) % args.log_interval == 0:
                emit({"event": "batch", "epoch": epoch + 1, "batch": batch_index + 1, "loss": loss_sum / samples, "lr": optimizer.param_groups[0]["lr"], "samples_per_rank": samples}, args.log_file, rank)
            if args.train_batches and batch_index + 1 >= args.train_batches:
                break
        scheduler.step()
        regular = evaluate(raw_model, val_loader, device, args.val_batches)
        ema_metrics = evaluate(ema.module, val_loader, device, args.val_batches)
        improved = ema_metrics["acc1"] > best_acc1
        best_acc1 = max(best_acc1, ema_metrics["acc1"])
        emit({"event": "epoch", "epoch": epoch + 1, "train_loss": loss_sum / samples, "seconds": round(time.time() - started, 3), "lr": optimizer.param_groups[0]["lr"], "model": regular, "ema": ema_metrics, "best_ema_acc1": best_acc1}, args.log_file, rank)
        if rank == 0:
            state = {"model": raw_model.state_dict(), "model_ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_acc1": best_acc1, "args": vars(args)}
            torch.save(state, Path(args.output_dir) / "last.pth")
            if improved:
                torch.save(state, Path(args.output_dir) / "best.pth")
        if args.smoke:
            break
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
