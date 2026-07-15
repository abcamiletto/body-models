"""Focused CUDA benchmark for SOMA ``forward_vertices`` inference and training.

Usage:
    uv run bench/soma_forward.py
    uv run bench/soma_forward.py --cuda-graph --mode forward --kernel torch
    uv run bench/soma_forward.py --batch-sizes 128,512,2048,4096,8192
    uv run bench/soma_forward.py --kernel warp --mode backward
"""

from __future__ import annotations

import argparse
import statistics

import torch

from body_models.bodies.soma import torch as soma_torch


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    kernels = args.kernels or ["torch", "warp"]
    modes = args.modes or ["forward", "backward"]
    print("| Kernel | Mode | Execution | Batch | Median (ms) | Peak allocated (GiB) |")
    print("|---|---|---|---:|---:|---:|")

    for kernel in kernels:
        model = soma_torch.SOMA(kernel=kernel).cuda().eval()
        for batch_size in batch_sizes:
            for mode in modes:
                executions = ["eager"]
                if args.cuda_graph and kernel == "torch" and mode == "forward":
                    executions.append("cuda-graph")
                for execution in executions:
                    try:
                        step = make_step(
                            model,
                            batch_size,
                            backward=mode == "backward",
                            cuda_graph=execution == "cuda-graph",
                        )
                        median_ms, peak_gib = benchmark(step, args.runs, args.warmup)
                        print(
                            f"| {kernel} | {mode} | {execution} | {batch_size} | {median_ms:.2f} | {peak_gib:.2f} |",
                            flush=True,
                        )
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"| {kernel} | {mode} | {execution} | {batch_size} | OOM | OOM |", flush=True)


def make_step(model: soma_torch.SOMA, batch_size: int, *, backward: bool, cuda_graph: bool = False):
    params = model.get_rest_pose(batch_dims=(batch_size,))
    identity = model.prepare_identity(params.pop("shape")[:1])

    if backward:
        params = {key: value.detach().requires_grad_(True) for key, value in params.items()}

        def step():
            for value in params.values():
                value.grad = None
            vertices = model.forward_vertices(**params, identity=identity)
            vertices.square().mean().backward()

        return step

    if cuda_graph:
        captured = model.capture_forward_vertices(**params, identity=identity)

        def step():
            captured(**params)

        return step

    def step():
        with torch.inference_mode():
            model.forward_vertices(**params, identity=identity)

    return step


def benchmark(step, runs: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return statistics.median(timings), peak_gib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8,32,128,256")
    parser.add_argument("--kernel", action="append", choices=soma_torch.SOMA.kernels, dest="kernels")
    parser.add_argument("--mode", action="append", choices=("forward", "backward"), dest="modes")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cuda-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
