#!/usr/bin/env python3
"""Train a YOLO26 model with Ultralytics on MetaX C500."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolo26n-objv1-150.pt", help="Official YOLO26 starting checkpoint (auto-downloaded by Ultralytics if absent)")
    p.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml",
                   help="Ultralytics dataset YAML")
    p.add_argument("--epochs", type=int, default=245)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0,1,2,3,4,5,6,7",
                   help="C500 device ids, comma-separated; defaults to all 8 cards")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=str(Path(__file__).parent / "runs"))
    p.add_argument("--name", default="yolo26n_official_recipe")
    p.add_argument("--resume", action="store_true", help="Resume from --model checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", action="store_true", help="Cache images (uses host memory/disk)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, project=args.project, name=args.name,
        resume=args.resume, seed=args.seed, cache=args.cache, optimizer="MuSGD",
        lr0=0.0054, lrf=0.04952, momentum=0.94676, weight_decay=0.00064,
        warmup_epochs=0.98124, warmup_momentum=0.6576, warmup_bias_lr=0.08114,
        nbs=64, close_mosaic=10, cos_lr=False, amp=True, deterministic=True,
        box=5.62767, cls=0.56099, dfl=9.03871,
        hsv_h=0.01373, hsv_s=0.64481, hsv_v=0.56565,
        degrees=1.11032, translate=0.07105, scale=0.56232,
        shear=1.46386, perspective=0.00011, flipud=0.05854,
        fliplr=0.60571, bgr=0.10567, mosaic=0.90863, mixup=0.01216,
        cutmix=0.0, copy_paste=0.07504, copy_paste_mode="flip",
        auto_augment="randaugment", erasing=0.4,
    )
    print("Training complete:", getattr(results, "save_dir", Path(args.project) / args.name))


if __name__ == "__main__":
    main()
