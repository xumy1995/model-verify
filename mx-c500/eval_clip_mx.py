import argparse
import glob
import os
import runpy
import time

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_PATH = "/mnt/afs/xumengying/models_and_datasets/clip-vit-base-patch32"
DEFAULT_DATASET_PATH = "/mnt/afs/xumengying/models_and_datasets/imagenet-1k"
DEFAULT_CACHE_DIR = "/mnt/afs/xumengying/hf_datasets_cache"

# Standard CLIP/ImageNet zero-shot prompt ensemble (the original paper uses
# this style of natural-language templates rather than classifier fine-tuning).
PROMPT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a photo of the cool {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a photo of my {}.",
    "a photo of your {}.",
    "a photo of a {} in the wild.",
    "a photo of a {} in nature.",
    "a photo of a {} outdoors.",
    "a photo of a {} indoors.",
)
SIMPLE_TEMPLATE = ("a photo of a {}.",)


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot CLIP evaluation on ImageNet-1k."
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--split", default="validation")
    p.add_argument(
        "--devices", nargs="+", default=["cpu", "cuda"], choices=["cpu", "cuda"]
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--text-batch-size", type=int, default=256)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--prompt-set", choices=["ensemble", "simple"], default="ensemble"
    )
    return p.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_dataset_split(path, split, cache):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        files = sorted(glob.glob(os.path.join(path, "data", f"{split}-*.parquet")))
        if files:
            return split, load_dataset(
                "parquet", data_files={split: files}, split=split, cache_dir=cache
            )
        return split, load_dataset(path, split=split, cache_dir=cache)
    except DatasetGenerationError as exc:
        raise RuntimeError(f"Failed to load ImageNet split {split}.") from exc


def class_names(dataset, dataset_path):
    try:
        feature = dataset.features["label"]
        if hasattr(feature, "names"):
            return list(feature.names)
    except (AttributeError, KeyError):
        pass
    classes = runpy.run_path(os.path.join(dataset_path, "classes.py"))[
        "IMAGENET2012_CLASSES"
    ]
    return list(classes.values())


def clean_name(name):
    return name.split(",")[0].replace("_", " ").strip()


def projected_features(output, projection):
    """Extract projected CLIP features across Transformers API versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        # Recent Transformers versions place the already projected feature in
        # ``pooler_output``; older versions may expose the projected tensor as
        # the direct return value (handled above).
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return projection(output.last_hidden_state[:, 0])
    raise TypeError(f"Unsupported CLIP feature output: {type(output)!r}")


def text_features(model, tokenizer, names, templates, device, batch_size, fp16):
    features = []
    with torch.inference_mode():
        for start in range(0, len(names), batch_size):
            batch_names = names[start : start + batch_size]
            texts = [
                template.format(clean_name(name))
                for name in batch_names
                for template in templates
            ]
            tokens = tokenizer(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            encoded = projected_features(
                model.get_text_features(**tokens), model.text_projection
            )
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            encoded = encoded.view(
                len(batch_names), len(templates), -1
            ).mean(dim=1)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded)
    return torch.cat(features, dim=0)


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    split, dataset = load_dataset_split(
        args.dataset_path, args.split, args.cache_dir
    )
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    names = class_names(dataset, args.dataset_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    templates = SIMPLE_TEMPLATE if args.prompt_set == "simple" else PROMPT_TEMPLATES
    print(f"Loaded split '{split}' with {len(dataset)} samples and {len(names)} classes.")

    for device_name in args.devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("\nRunning on: cuda\nCUDA not available, skipped.")
            continue
        device = torch.device(device_name)
        fp16 = args.fp16 and device.type == "cuda"
        start = time.perf_counter()
        model = CLIPModel.from_pretrained(args.model_path).eval().to(device)
        if fp16:
            model.half()
        sync(device)
        load_time = time.perf_counter() - start
        text = text_features(
            model, processor, names, templates, device, args.text_batch_size, fp16
        )

        def collate(batch):
            pixels = processor(
                images=[item["image"].convert("RGB") for item in batch],
                return_tensors="pt",
            )["pixel_values"]
            labels = torch.tensor(
                [item.get("label", -1) for item in batch], dtype=torch.long
            )
            return pixels, labels

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        c1 = c5 = total = measured = 0
        infer_time = wall = prep_time = 0.0
        for i, (pixels, labels) in tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"{device} {split} eval",
            unit="batch",
        ):
            measured_batch = i >= args.warmup_batches
            t0 = time.perf_counter()
            tprep = time.perf_counter()
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if fp16:
                pixels = pixels.half()
            sync(device)
            prep = time.perf_counter() - tprep
            ti = time.perf_counter()
            with torch.inference_mode():
                image = projected_features(
                    model.get_image_features(pixel_values=pixels),
                    model.visual_projection,
                )
                image = image / image.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * image @ text.T
            sync(device)
            inf = time.perf_counter() - ti
            if measured_batch:
                measured += labels.numel()
                wall += time.perf_counter() - t0
                prep_time += prep
                infer_time += inf
            valid = labels >= 0
            pred = logits.argmax(1)
            c1 += (pred[valid] == labels[valid]).sum().item()
            top5_indices = logits[valid].topk(min(5, logits.shape[1]), 1).indices
            c5 += (top5_indices == labels[valid, None]).any(1).sum().item()
            total += labels.numel()
        n = total
        print(
            f"\nRunning on: {device}\n"
            f"Model path: {args.model_path}\n"
            f"Samples: {n}\n"
            f"Prompt set: {args.prompt_set} ({len(templates)} templates)\n"
            f"Model load time: {load_time:.3f}s\n"
            f"Preprocess time: {prep_time:.3f}s\n"
            f"Inference time: {infer_time:.3f}s\n"
            f"Throughput: {measured / wall if wall else 0:.2f} samples/s\n"
            f"Infer throughput: {measured / infer_time if infer_time else 0:.2f} samples/s\n"
            f"Avg infer latency: {1000 * infer_time / measured if measured else 0:.3f} ms/sample\n"
            f"Top-1 accuracy: {100 * c1 / n:.2f}%\n"
            f"Top-5 accuracy: {100 * c5 / n:.2f}%"
        )


if __name__ == "__main__":
    main()
