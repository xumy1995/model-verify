#!/usr/bin/env python3
"""Train a YOLO26 model with Ultralytics on MetaX C500."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26n-objv1-150.pt", help="Official YOLO26 starting checkpoint")
    p.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml",
                   help="Ultralytics dataset YAML")
    p.add_argument("--epochs", type=int, default=100)
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
    )
    print("Training complete:", getattr(results, "save_dir", Path(args.project) / args.name))


if __name__ == "__main__":
    main()
