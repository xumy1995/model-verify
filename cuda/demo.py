from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import time


model_path = "/data/xumengying/models_and_datasets/resnet-50"
dataset_path = "/data/xumengying/models_and_datasets/cats-image"


# -------------------------
# Load dataset
# -------------------------
dataset = load_dataset(dataset_path)
image = dataset["test"]["image"][0]


# -------------------------
# Load processor
# -------------------------
processor = AutoImageProcessor.from_pretrained(model_path)


def run(device):
    print("\n========================")
    print("Running on:", device)
    print("========================")

    # load model
    start = time.perf_counter()

    model = ResNetForImageClassification.from_pretrained(
        model_path
    )

    model = model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()

    load_time = time.perf_counter() - start


    print(
        "Model device:",
        next(model.parameters()).device
    )


    # -------------------------
    # preprocess
    # -------------------------
    start = time.perf_counter()

    inputs = processor(
        image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    if device == "cuda":
        torch.cuda.synchronize()

    preprocess_time = time.perf_counter() - start


    # -------------------------
    # inference
    # -------------------------
    start = time.perf_counter()

    with torch.no_grad():
        logits = model(**inputs).logits

    if device == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start


    predicted_label = logits.argmax(-1).item()

    print(f"Load model time : {load_time:.6f}s")
    print(f"Preprocess time : {preprocess_time:.6f}s")
    print(f"Inference time  : {inference_time:.6f}s")
    print(f"Total time      : {load_time + preprocess_time + inference_time:.6f}s")

    print(
        "Prediction:",
        model.config.id2label[predicted_label]
    )


# -------------------------
# CPU
# -------------------------
run("cpu")


# -------------------------
# CUDA
# -------------------------
if torch.cuda.is_available():
    run("cuda")
else:
    print("CUDA not available")