"""Measure inference only; no accuracy claim is made from synthetic inputs."""
import platform
import statistics
import time
from pathlib import Path
import json
import hashlib

import torch

from .train import choose_device, load_model


def benchmark(checkpoint, output, device='auto', batch_size=1, warmup=5, repetitions=30):
    if batch_size < 1 or warmup < 0 or repetitions < 1:
        raise ValueError('batch_size and repetitions must be positive; warmup must be nonnegative')
    path = Path(output)
    if path.exists():
        raise ValueError('Benchmark output exists')
    selected = choose_device(device)
    model, payload = load_model(checkpoint, selected)
    size = payload['config']['image_size']
    inputs = torch.zeros(batch_size, 3, size, size, device=selected)
    def synchronize():
        if selected.type == 'cuda':
            torch.cuda.synchronize(selected)
    if selected.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(selected)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        synchronize()
        for _ in range(repetitions):
            start = time.perf_counter()
            model(inputs)
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
    result = {'checkpoint':str(Path(checkpoint).resolve()), 'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(), 'device':str(selected),
              'hardware':torch.cuda.get_device_name(selected) if selected.type == 'cuda' else platform.processor(),
              'torch_version':torch.__version__, 'cuda_runtime':torch.version.cuda,
              'image_size':size, 'batch_size':batch_size, 'warmup':warmup, 'repetitions':repetitions,
              'latency_ms_mean_per_batch':statistics.mean(times),
              'latency_ms_median_per_batch':statistics.median(times),
              'peak_gpu_bytes':torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None,
              'note':'Forward pass on synthetic tensors; excludes loading and preprocessing; no accuracy evaluation.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
