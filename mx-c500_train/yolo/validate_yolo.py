#!/usr/bin/env python3
"""Validate a trained YOLO26 checkpoint on C500."""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to best.pt/last.pt")
    p.add_argument("--data", default="/mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    metrics = YOLO(args.model).val(data=args.data, split="val", device=args.device,
                                  batch=args.batch, imgsz=args.imgsz)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP75:    {metrics.box.map75:.4f}")
    print("per-class:", metrics.box.maps)


if __name__ == "__main__":
    main()
