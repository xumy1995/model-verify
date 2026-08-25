#!/usr/bin/env python3
"""Operator and stage profiler for ViT on MX-C500."""

import argparse
import os
import time

import torch
from transformers import ViTForImageClassification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-path',
        default='/mnt/afs/xumengying/models_and_datasets/vit-base-patch16-224',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--trace', default='')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    device = torch.device(args.device)
    model = ViTForImageClassification.from_pretrained(args.model_path)
    model = model.eval().to(device)
    if args.fp16:
        model.half()
    inputs = torch.randn(args.batch_size, 3, 224, 224, device=device)
    if args.fp16:
        inputs = inputs.half()

    named = dict(model.vit.named_modules())
    stages = []
    embedding = next(
        (m for n, m in named.items() if n.endswith('embeddings')), None
    )
    if embedding is not None:
        stages.append(('embeddings', embedding))
    blocks = [
        (n, m) for n, m in named.items()
        if n and any(k in n for k in ('layer.', 'layers.', 'blocks.'))
        and len(list(m.children())) > 0
        and n.rsplit('.', 1)[-1].isdigit()
    ]
    stages.extend((f'encoder.{n}', m) for n, m in blocks)
    norm = next(
        (m for n, m in named.items()
         if n.endswith('layernorm') or n.endswith('post_layernorm')),
        None,
    )
    if norm is not None:
        stages.append(('layernorm', norm))
    stages.append(('classifier', model.classifier))

    totals = {name: 0.0 for name, _ in stages}
    starts = {}

    def before(name):
        def hook(_module, _inputs):
            torch.cuda.synchronize(device)
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(_module, _inputs, _output):
            torch.cuda.synchronize(device)
            totals[name] += (time.perf_counter() - starts.pop(name)) * 1000
        return hook

    handles = []
    for name, module in stages:
        handles.append(module.register_forward_pre_hook(before(name)))
        handles.append(module.register_forward_hook(after(name)))
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(pixel_values=inputs)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(args.steps):
                model(pixel_values=inputs)
                torch.cuda.synchronize(device)
                profiler.step()
    for handle in handles:
        handle.remove()
    print('Stage wall time (average ms per step):')
    for name, _ in stages:
        print(f'{name:24s} {totals[name] / args.steps:10.3f}')
    print(profiler.key_averages().table(sort_by='cuda_time_total', row_limit=40))
    if args.trace:
        os.makedirs(os.path.dirname(os.path.abspath(args.trace)), exist_ok=True)
        profiler.export_chrome_trace(args.trace)


if __name__ == '__main__':
    main()
