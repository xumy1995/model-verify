"""Generate the CUDA/MX-C500 comparison table from evaluation logs."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def number(pattern, text):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def fmt(value, digits=4):
    return "未记录" if value is None else f"{value:.{digits}f}"


def comparison(label, cuda, mx):
    if cuda is None or mx is None:
        return "未记录"
    accuracy = any(k in label.lower() for k in ("accuracy", "ap", "map"))
    if accuracy:
        diff = mx - cuda
        value = f"{diff:+.2f} %"
        return f"**{value}**" if diff <= -5.0 else value
    # For throughput, MX/CUDA is the performance ratio. For latency, lower is
    # better, so report CUDA-time / MX-time (the effective performance ratio).
    is_latency = "耗时" in label or "latency" in label.lower()
    ratio = (100.0 * cuda / mx if is_latency else 100.0 * mx / cuda) if cuda and mx else None
    value = f"{ratio:.2f}%"
    return f"**{value}**" if ratio < 80.0 else value


def is_accuracy(label):
    return any(k in label.lower() for k in ("accuracy", "ap", "map"))


def classification_rows(model, dataset, cuda_file, mx_file):
    rows = []
    for label, pattern, digits in (
        ("Top-1 accuracy (%)", r"Top-1 accuracy\s*:\s*([0-9.]+)%", 2),
        ("Top-5 accuracy (%)", r"Top-5 accuracy\s*:\s*([0-9.]+)%", 2),
    ):
        rows.append((model, dataset, label, number(pattern, cuda_file), number(pattern, mx_file), digits))
    # Throughput is samples/s; its reciprocal is ms/sample.
    for label, pattern, digits in (
        ("推理耗时", r"Infer throughput\s*:\s*([0-9.]+)", 4),
        ("端到端耗时", r"Throughput\s*:\s*([0-9.]+)", 4),
    ):
        c, m = number(pattern, cuda_file), number(pattern, mx_file)
        rows.append((model, dataset, label, 1000 / c if c else None, 1000 / m if m else None, digits))
    return rows


def detr_rows(cuda_file, mx_file):
    rows = []
    for label, pattern in (
        # Anchor on the AP label; otherwise the same IoU/area pattern also
        # matches the later AR summary line and number() returns AR@100.
        ("AP (IoU 0.50:0.95)", r"Average Precision\s+\(AP\).*IoU=0.50:0.95 \| area=\s+all.*=\s*([0-9.]+)"),
        ("AP50", r"Average Precision\s+\(AP\).*IoU=0.50\s+\| area=\s+all.*=\s*([0-9.]+)"),
        ("AP75", r"Average Precision\s+\(AP\).*IoU=0.75\s+\| area=\s+all.*=\s*([0-9.]+)"),
    ):
        rows.append(("DETR-ResNet-50", "COCO val", label, number(pattern, cuda_file), number(pattern, mx_file), 3))
    for label, pattern in (("推理耗时", r"Avg infer latency:\s*([0-9.]+)"), ("端到端耗时", r"Throughput\s*:\s*([0-9.]+)")):
        c, m = number(pattern, cuda_file), number(pattern, mx_file)
        rows.append(("DETR-ResNet-50", "COCO val", label, c if "latency" in pattern else (1000 / c if c else None), m if "latency" in pattern else (1000 / m if m else None), 3))
    return rows


def yolo_rows(model, cuda_file, mx_file):
    rows = []
    for label, key in (("mAP50-95", "mAP50-95"), ("mAP50", "mAP50"), ("mAP75", "mAP75")):
        rows.append((model, "COCO val", label, number(rf"{key}:\s*([0-9.]+)", cuda_file), number(rf"{key}:\s*([0-9.]+)", mx_file), 4))
    for label in ("推理耗时", "端到端耗时"):
        vals = []
        for text in (cuda_file, mx_file):
            parts = [number(rf"{key}:\s*([0-9.]+)", text) for key in ("inference_speed", "preprocess_speed", "postprocess_speed")]
            vals.append(parts[0] if "推理" in label else (sum(x for x in parts if x is not None) if all(x is not None for x in parts) else None))
        rows.append((model, "COCO val", label, vals[0], vals[1], 4))
    return rows


def main():
    rows = []
    for model, stem, dataset in (("ResNet-50", "resnet50-imagenet-1k-val", "ImageNet-1k val"), ("ViT-B/16", "VIT-B-16-imagenet-1k-val", "ImageNet-1k val"), ("CLIP ViT-B/32 (zero-shot)", "clip-vit-b-32-imagenet-1k-val", "ImageNet-1k val")):
        c = (ROOT / "cuda/logs" / f"{stem}-cuda.log").read_text(errors="ignore")
        m = (ROOT / "mx-c500/logs" / f"{stem}-mx-c500.log").read_text(errors="ignore")
        rows += classification_rows(model, dataset, c, m)
    c = (ROOT / "cuda/logs/detr-resnet-50-coco-val-cuda.log").read_text(errors="ignore")
    m = (ROOT / "mx-c500/logs/detr-resnet-50-coco-val-mx-c500.log").read_text(errors="ignore")
    rows += detr_rows(c, m)
    for size in ("n", "s", "m", "l", "x"):
        c = (ROOT / "cuda/logs" / f"yolo26{size}-coco-val-cuda.log").read_text(errors="ignore")
        m = (ROOT / "mx-c500/logs" / f"yolo26{size}-coco-val-mx-c500.log").read_text(errors="ignore")
        rows += yolo_rows(f"YOLO26{size}", c, m)
    for title, selected in (
        ("精度对比", [row for row in rows if is_accuracy(row[2])]),
        ("性能对比", [row for row in rows if not is_accuracy(row[2])]),
    ):
        print(f"### {title}\n")
        if title == "性能对比":
            print(f"单位: ms/sample\n")
        print("| 模型 | 数据集 | 评测项 | cuda指标值 | mx-c500指标值 | 对比结果 |")
        print("|---|---|---|---:|---:|---:|")
        for model, dataset, label, c, m, digits in selected:
            print(f"| {model} | {dataset} | {label} | {fmt(c, digits)} | {fmt(m, digits)} | {comparison(label, c, m)} |")
        print()


if __name__ == "__main__":
    main()
