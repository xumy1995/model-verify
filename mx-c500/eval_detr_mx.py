import argparse
import glob
import io
import json
import os
import time

# Keep older onnx/protobuf installations importable through transformers.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from datasets import DatasetDict, load_dataset
from datasets.exceptions import DatasetGenerationError
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, DetrForObjectDetection


DEFAULT_MODEL_PATH = "/mnt/afs/xumengying/models_and_datasets/detr-resnet-50"
DEFAULT_DATASET_PATH = "/mnt/afs/xumengying/models_and_datasets/coco"
DEFAULT_CACHE_DIR = "/mnt/afs/xumengying/hf_datasets_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DETR accuracy and performance on COCO."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate. Use 'auto' for validation/val/test fallback.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda"],
        choices=["cpu", "cuda"],
        help="Devices to test.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 evaluates the whole split."
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=2,
        help="Warmup batches excluded from performance metrics.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Score threshold before COCO evaluation. Keep 0 for standard AP.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16 on CUDA.")
    return parser.parse_args()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_coco_dataset(dataset_path, split, cache_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    candidates = ("validation", "val", "test") if split == "auto" else (split,)
    try:
        for candidate in candidates:
            files = sorted(
                glob.glob(os.path.join(dataset_path, "data", f"{candidate}-*.parquet"))
            )
            if files:
                dataset = load_dataset(
                    "parquet",
                    data_files={candidate: files},
                    split=candidate,
                    cache_dir=cache_dir,
                )
                return candidate, dataset
            try:
                dataset = load_dataset(
                    dataset_path, split=candidate, cache_dir=cache_dir
                )
                return candidate, dataset
            except (DatasetGenerationError, ValueError, KeyError):
                if split != "auto":
                    raise
    except DatasetGenerationError as exc:
        raise RuntimeError(
            f"Failed to load COCO from {dataset_path}; verify all parquet shards exist."
        ) from exc

    dataset = load_dataset(dataset_path, cache_dir=cache_dir)
    if not isinstance(dataset, DatasetDict):
        return "default", dataset
    for candidate in candidates:
        if candidate in dataset:
            return candidate, dataset[candidate]
    raise RuntimeError(f"No evaluation split found under {dataset_path}")


def category_names(dataset, dataset_path):
    try:
        objects_feature = dataset.features["objects"]
        if hasattr(objects_feature, "feature"):
            objects_feature = objects_feature.feature
        category_feature = objects_feature["category"]
        while hasattr(category_feature, "feature"):
            category_feature = category_feature.feature
        return category_feature.names
    except (AttributeError, KeyError):
        info_path = os.path.join(dataset_path, "dataset_infos.json")
        if os.path.isfile(info_path):
            with open(info_path, encoding="utf-8") as info_file:
                infos = json.load(info_file)
            for info in infos.values():
                category = info.get("features", {}).get("objects", {}).get(
                    "feature", {}
                ).get("category", {})
                if category.get("names"):
                    return category["names"]
        raise RuntimeError("Dataset category names are unavailable; cannot map labels.")


def image_from_value(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def collate_images(processor):
    def collate(batch):
        images = [image_from_value(item["image"]) for item in batch]
        inputs = processor(images=images, return_tensors="pt")
        metadata = [
            {
                "image_id": int(item["image_id"]),
                "height": int(item.get("height", image.height)),
                "width": int(item.get("width", image.width)),
                "objects": item["objects"],
            }
            for item, image in zip(batch, images)
        ]
        return inputs, metadata

    return collate


def build_ground_truth(metadata, names):
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(names)]
    images = []
    annotations = []
    annotation_id = 1
    for item in metadata:
        images.append(
            {"id": item["image_id"], "height": item["height"], "width": item["width"]}
        )
        objects = item["objects"]
        for category, bbox, area in zip(
            objects["category"], objects["bbox"], objects["area"]
        ):
            x1, y1, x2, y2 = bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": item["image_id"],
                    "category_id": int(category) + 1,
                    "bbox": [float(x1), float(y1), width, height],
                    "area": float(area) if area is not None else width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def model_to_dataset_category_ids(model, names):
    name_to_id = {name: index + 1 for index, name in enumerate(names)}
    mapping = {}
    for model_id, label in model.config.id2label.items():
        if label in name_to_id:
            mapping[int(model_id)] = name_to_id[label]
    missing = set(names) - set(model.config.id2label.values())
    if missing:
        raise RuntimeError(f"Model is missing dataset categories: {sorted(missing)}")
    return mapping


def evaluate_coco(ground_truth, predictions):
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    if not predictions:
        print("No detections survived the score threshold; COCO AP is 0.")
        return
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [image["id"] for image in ground_truth["images"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()


def run_eval(args, split_name, dataset, processor, names, device_name):
    print("\n========================")
    print(f"Running on: {device_name}")
    print("========================")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipped.")
        return

    device = torch.device(device_name)
    use_fp16 = args.fp16 and device.type == "cuda"
    start = time.perf_counter()
    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.eval().to(device)
    if use_fp16:
        model.half()
    category_mapping = model_to_dataset_category_ids(model, names)
    sync_if_cuda(device)
    load_time = time.perf_counter() - start

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_images(processor),
    )
    predictions = []
    all_metadata = []
    total_seen = measured_seen = 0
    preprocess_time = inference_time = measured_wall_time = 0.0
    progress = tqdm(
        enumerate(loader), total=len(loader), desc=f"{device} {split_name} eval", unit="batch"
    )

    with torch.inference_mode():
        for batch_index, (inputs, metadata) in progress:
            measured = batch_index >= args.warmup_batches
            batch_start = time.perf_counter()
            prep_start = time.perf_counter()
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
            pixel_mask = inputs.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            if use_fp16:
                pixel_values = pixel_values.half()
            sync_if_cuda(device)
            prep_elapsed = time.perf_counter() - prep_start

            infer_start = time.perf_counter()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            sync_if_cuda(device)
            infer_elapsed = time.perf_counter() - infer_start
            if measured:
                measured_wall_time += time.perf_counter() - batch_start
                preprocess_time += prep_elapsed
                inference_time += infer_elapsed
                measured_seen += len(metadata)

            target_sizes = torch.tensor(
                [[item["height"], item["width"]] for item in metadata], device=device
            )
            results = processor.post_process_object_detection(
                outputs, threshold=args.threshold, target_sizes=target_sizes
            )
            for item, result in zip(metadata, results):
                for score, label, box in zip(
                    result["scores"].cpu(), result["labels"].cpu(), result["boxes"].cpu()
                ):
                    model_label = int(label)
                    if model_label not in category_mapping:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append(
                        {
                            "image_id": item["image_id"],
                            "category_id": category_mapping[model_label],
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": float(score),
                        }
                    )
            all_metadata.extend(metadata)
            total_seen += len(metadata)
            postfix = {"samples": total_seen, "detections": len(predictions)}
            if measured_wall_time > 0:
                postfix["img/s"] = f"{measured_seen / measured_wall_time:.2f}"
            progress.set_postfix(postfix)

    throughput = measured_seen / measured_wall_time if measured_wall_time else 0.0
    infer_throughput = measured_seen / inference_time if inference_time else 0.0
    latency = 1000.0 * inference_time / measured_seen if measured_seen else 0.0
    print(f"Model path       : {args.model_path}")
    print(f"Dataset path     : {args.dataset_path}")
    print(f"Split            : {split_name}")
    print(f"Samples          : {total_seen}")
    print(f"Detections       : {len(predictions)}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Score threshold  : {args.threshold}")
    print(f"Precision        : {'fp16' if use_fp16 else 'fp32'}")
    print(f"Model load time  : {load_time:.3f}s")
    print(f"Preprocess time  : {preprocess_time:.3f}s")
    print(f"Inference time   : {inference_time:.3f}s")
    print(f"Measured wall    : {measured_wall_time:.3f}s")
    print(f"Throughput       : {throughput:.2f} samples/s")
    print(f"Infer throughput : {infer_throughput:.2f} samples/s")
    print(f"Avg infer latency: {latency:.3f} ms/sample")
    evaluate_coco(build_ground_truth(all_metadata, names), predictions)


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    split_name, dataset = load_coco_dataset(args.dataset_path, args.split, args.cache_dir)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if not len(dataset):
        raise RuntimeError("Dataset split is empty.")
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    names = category_names(dataset, args.dataset_path)
    print(f"Loaded split '{split_name}' with {len(dataset)} samples and {len(names)} classes.")
    for device_name in args.devices:
        run_eval(args, split_name, dataset, processor, names, device_name)


if __name__ == "__main__":
    main()
