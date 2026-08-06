from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/afs/xumengying/models_and_datasets/YOLO26/yolo26n.pt",
        help="Path to YOLO model (.pt)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id"
    )

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Validate
    metrics = model.val(
        data="/mnt/afs/xumengying/models_and_datasets/coco_yolo_format/coco.yaml",
        split="val",
        device=args.device
    )

    print("Model:", args.model)
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("per class:", metrics.box.maps)
    print("preprocess_speed:", metrics.speed["preprocess"])
    print("inference_speed:", metrics.speed["inference"])
    print("postprocess_speed:", metrics.speed["postprocess"])


if __name__ == "__main__":
    main()