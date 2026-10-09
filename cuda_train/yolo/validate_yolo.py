#!/usr/bin/env python3
"""Evaluate a trained YOLO26 checkpoint on COCO validation images."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent / "runs/yolo26n_official_recipe/weights/best.pt"),
        help="Trained best.pt or last.pt checkpoint",
    )
    parser.add_argument(
        "--data",
        default="/data/xumengying/models_and_datasets/coco_yolo_format/coco.yaml",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default=str(Path(__file__).parent / "runs"))
    parser.add_argument("--name", default="validation_best")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = YOLO(args.model).val(
        data=args.data, split="val", device=args.device, batch=args.batch, imgsz=args.imgsz,
        project=args.project, name=args.name, exist_ok=True,
    )
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    print("per-class:", metrics.box.maps)


if __name__ == "__main__":
    main()
