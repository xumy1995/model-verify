import argparse
import glob
import os
import time

# Keep older onnx/protobuf installations importable through transformers.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from datasets import DatasetDict, load_dataset
from datasets.exceptions import DatasetGenerationError
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification


DEFAULT_MODEL_PATH = "/mnt/afs/xumengying/models_and_datasets/vit-base-patch16-224"
DEFAULT_DATASET_PATH = "/mnt/afs/xumengying/models_and_datasets/imagenet-1k"
DEFAULT_CACHE_DIR = "/mnt/afs/xumengying/hf_datasets_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate google/vit-base-patch16-224 accuracy and performance "
            "on ImageNet-1k."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate. Use 'auto' for validation/val/test fallback.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="Devices to test.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples. 0 means use the whole split.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Warmup batches excluded from performance metrics.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 on CUDA. CPU always uses float32.",
    )
    return parser.parse_args()


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_imagenet_dataset(dataset_path, split, cache_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    try:
        if split != "auto":
            parquet_pattern = os.path.join(dataset_path, "data", f"{split}-*.parquet")
            parquet_files = sorted(glob.glob(parquet_pattern))
            if parquet_files:
                return split, load_dataset(
                    "parquet",
                    data_files={split: parquet_files},
                    split=split,
                    cache_dir=cache_dir,
                )
            return split, load_dataset(dataset_path, split=split, cache_dir=cache_dir)

        for candidate in ("validation", "val", "test"):
            parquet_pattern = os.path.join(
                dataset_path, "data", f"{candidate}-*.parquet"
            )
            parquet_files = sorted(glob.glob(parquet_pattern))
            if parquet_files:
                return candidate, load_dataset(
                    "parquet",
                    data_files={candidate: parquet_files},
                    split=candidate,
                    cache_dir=cache_dir,
                )
            try:
                return candidate, load_dataset(
                    dataset_path, split=candidate, cache_dir=cache_dir
                )
            except (DatasetGenerationError, ValueError, KeyError):
                continue
        dataset = load_dataset(dataset_path, cache_dir=cache_dir)
    except DatasetGenerationError as exc:
        raise RuntimeError(
            "Failed to load ImageNet-1k. The dataset looks incomplete or a parquet "
            f"shard is still downloading under {dataset_path}. Wait for the download "
            "to finish, then run this script again."
        ) from exc

    if not isinstance(dataset, DatasetDict):
        return "default", dataset

    for candidate in ("validation", "val", "test"):
        if candidate in dataset:
            return candidate, dataset[candidate]

    available = list(dataset.keys())
    if not available:
        raise RuntimeError(f"No dataset splits found under {dataset_path}")
    return available[0], dataset[available[0]]


def collate_images(processor):
    def collate(batch):
        images = [item["image"].convert("RGB") for item in batch]
        labels = torch.tensor(
            [item.get("label", -1) for item in batch], dtype=torch.long
        )
        inputs = processor(images=images, return_tensors="pt")
        return inputs["pixel_values"], labels

    return collate


def compute_correct(logits, labels):
    valid = labels >= 0
    if not valid.any():
        return 0, 0, 0

    valid_logits = logits[valid]
    valid_labels = labels[valid]
    top1 = valid_logits.argmax(dim=1)
    top1_correct = (top1 == valid_labels).sum().item()

    topk = min(5, valid_logits.shape[1])
    top5 = valid_logits.topk(topk, dim=1).indices
    top5_correct = (top5 == valid_labels.unsqueeze(1)).any(dim=1).sum().item()
    return top1_correct, top5_correct, valid_labels.numel()


def run_eval(args, split_name, dataset, processor, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("\n========================")
        print("Running on: cuda")
        print("========================")
        print("CUDA not available, skipped.")
        return

    device = torch.device(device_name)
    use_fp16 = args.fp16 and device.type == "cuda"

    print("\n========================")
    print(f"Running on: {device}")
    print("========================")

    start = time.perf_counter()
    model = ViTForImageClassification.from_pretrained(args.model_path)
    model.eval()
    model.to(device)
    if use_fp16:
        model.half()
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

    total_seen = 0
    measured_seen = 0
    labeled_seen = 0
    top1_correct = 0
    top5_correct = 0
    preprocess_time = 0.0
    inference_time = 0.0
    measured_wall_time = 0.0

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"{device} {split_name} eval",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    with torch.inference_mode():
        for batch_index, (pixel_values, labels) in progress:
            batch_size = labels.numel()
            measured = batch_index >= args.warmup_batches
            batch_start = time.perf_counter()

            prep_start = time.perf_counter()
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_fp16:
                pixel_values = pixel_values.half()
            sync_if_cuda(device)
            prep_elapsed = time.perf_counter() - prep_start

            infer_start = time.perf_counter()
            logits = model(pixel_values=pixel_values).logits
            sync_if_cuda(device)
            infer_elapsed = time.perf_counter() - infer_start

            if measured:
                measured_wall_time += time.perf_counter() - batch_start
                preprocess_time += prep_elapsed
                inference_time += infer_elapsed
                measured_seen += batch_size

            correct1, correct5, valid_count = compute_correct(logits, labels)
            top1_correct += correct1
            top5_correct += correct5
            labeled_seen += valid_count
            total_seen += batch_size

            postfix = {"samples": total_seen}
            if measured_wall_time > 0:
                postfix["img/s"] = f"{measured_seen / measured_wall_time:.2f}"
            if labeled_seen > 0:
                postfix["top1"] = f"{100.0 * top1_correct / labeled_seen:.2f}%"
            progress.set_postfix(postfix)

    accuracy_available = labeled_seen > 0
    top1 = 100.0 * top1_correct / labeled_seen if accuracy_available else None
    top5 = 100.0 * top5_correct / labeled_seen if accuracy_available else None
    throughput = measured_seen / measured_wall_time if measured_wall_time > 0 else 0.0
    inference_throughput = measured_seen / inference_time if inference_time > 0 else 0.0
    avg_latency_ms = (
        1000.0 * inference_time / measured_seen if measured_seen > 0 else 0.0
    )

    print(f"Model path       : {args.model_path}")
    print(f"Dataset path     : {args.dataset_path}")
    print(f"Split            : {split_name}")
    print(f"Samples          : {total_seen}")
    print(f"Labeled samples  : {labeled_seen}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Warmup batches   : {args.warmup_batches}")
    print(f"Precision        : {'fp16' if use_fp16 else 'fp32'}")
    print(f"Model load time  : {load_time:.3f}s")
    print(f"Preprocess time  : {preprocess_time:.3f}s")
    print(f"Inference time   : {inference_time:.3f}s")
    print(f"Measured wall    : {measured_wall_time:.3f}s")
    print(f"Throughput       : {throughput:.2f} samples/s")
    print(f"Infer throughput : {inference_throughput:.2f} samples/s")
    print(f"Avg infer latency: {avg_latency_ms:.3f} ms/sample")

    if accuracy_available:
        print(f"Top-1 accuracy   : {top1:.2f}%")
        print(f"Top-5 accuracy   : {top5:.2f}%")
    else:
        print("Top-1 accuracy   : N/A (labels are missing or set to -1)")
        print("Top-5 accuracy   : N/A (labels are missing or set to -1)")


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)

    split_name, dataset = load_imagenet_dataset(
        args.dataset_path,
        args.split,
        args.cache_dir,
    )
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    processor = AutoImageProcessor.from_pretrained(args.model_path)

    print(f"Loaded split '{split_name}' with {len(dataset)} samples.")
    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset split is empty. Wait for the ImageNet-1k download to finish."
        )

    for device_name in args.devices:
        run_eval(args, split_name, dataset, processor, device_name)


if __name__ == "__main__":
    main()
