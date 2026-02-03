#!/usr/bin/env python3
"""Benchmark body model forward and backward passes.

Measures forward_vertices, forward_skeleton, and their backward passes with
proper warmup, outlier rejection (IQR method), and gradient flow verification.

Examples:
    # Basic benchmark
    uv run scripts/benchmark.py --model mhr

    # Compare eager vs compiled performance
    uv run scripts/benchmark.py --model mhr --compile --compile-autotune

    # Custom batch sizes and longer measurement
    uv run scripts/benchmark.py --model anny --batch-sizes 1,16,64 --min-run-time 5

Available models: anny, mhr, skel, smpl, smplx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch._dynamo
import torch.utils.benchmark as benchmark
from torch import Tensor

torch._dynamo.config.suppress_errors = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from body_models import ANNY, MHR, SKEL, SMPL, SMPLX  # noqa: E402

MODELS = {"anny": ANNY, "mhr": MHR, "skel": SKEL, "smpl": SMPL, "smplx": SMPLX}
MODEL_KWARGS = {
    "anny": {},
    "mhr": {},
    "skel": {"gender": "male"},
    "smpl": {"gender": "neutral"},
    "smplx": {"gender": "neutral"},
}
COMPILE_MODES = {
    "default": {},
    "max-autotune": {"mode": "max-autotune"},
    "reduce-overhead": {"mode": "reduce-overhead"},
}


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    mean_ms: float
    std_ms: float
    median_ms: float
    iqr_ms: float


def add_noise_to_params(params: dict[str, Tensor], seed: int, noise_scale: float = 0.1) -> dict[str, Tensor]:
    """Add small noise to pose parameters for realistic benchmark conditions."""
    torch.manual_seed(seed)
    return {
        k: (v + torch.randn_like(v) * noise_scale) if v.dtype.is_floating_point and k != "shape" else v.clone()
        for k, v in params.items()
    }


def compute_robust_stats(times_ms: list[float]) -> tuple[float, float, float, float]:
    """Compute mean, std (with outlier rejection), median, and IQR."""
    times_sorted = sorted(times_ms)
    n = len(times_sorted)

    median = times_sorted[n // 2]
    q1 = times_sorted[n // 4] if n >= 4 else times_sorted[0]
    q3 = times_sorted[3 * n // 4] if n >= 4 else times_sorted[-1]
    iqr = q3 - q1

    # Reject outliers (outside 1.5 * IQR)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [t for t in times_ms if lower <= t <= upper] or times_ms

    mean = sum(filtered) / len(filtered)
    std = (sum((t - mean) ** 2 for t in filtered) / len(filtered)) ** 0.5

    return mean, std, median, iqr


def benchmark_fn(fn, desc: str, min_run_time: float) -> BenchmarkResult:
    """Benchmark a function with proper warmup and outlier rejection."""
    timer = benchmark.Timer(stmt="fn()", globals={"fn": fn}, description=desc, num_threads=1)
    measurement = timer.blocked_autorange(min_run_time=min_run_time)
    times_ms = [t * 1000 for t in measurement.times]
    mean, std, median, iqr = compute_robust_stats(times_ms)
    return BenchmarkResult(name=desc, batch_size=0, mean_ms=mean, std_ms=std, median_ms=median, iqr_ms=iqr)


def verify_gradients_flow(model, params: dict[str, Tensor], forward_method: str, device: str) -> None:
    """Verify gradients flow to at least one input parameter."""
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    output = getattr(model, forward_method)(**params)
    output.sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()

    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params.values()):
        raise RuntimeError(f"No gradients computed for {forward_method}!")

    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()


def run_benchmark(
    model_name: str,
    batch_sizes: list[int],
    compile_mode: str | None,
    device: str,
    dtype: torch.dtype,
    seed: int,
    min_run_time: float,
) -> list[BenchmarkResult]:
    """Run benchmark for a model with various batch sizes."""
    model = MODELS[model_name](**MODEL_KWARGS[model_name]).eval().to(device=device, dtype=dtype)

    if compile_mode:
        print(f"Compiling with mode={compile_mode}...")
        t0 = time.perf_counter()
        model = torch.compile(model, **COMPILE_MODES[compile_mode])

        # Warmup with ALL batch sizes to avoid recompilation during benchmark
        for bs in batch_sizes:
            warmup_params = {
                k: v.to(device=device, dtype=dtype) for k, v in model.get_rest_pose(batch_size=bs, dtype=dtype).items()
            }
            with torch.no_grad():
                model.forward_vertices(**warmup_params)
                model.forward_skeleton(**warmup_params)
            if device == "cuda":
                torch.cuda.synchronize()
        print(f"Compilation took {time.perf_counter() - t0:.1f}s")

    results = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")

        params = {
            k: v.to(device=device, dtype=dtype)
            for k, v in add_noise_to_params(model.get_rest_pose(batch_size=batch_size, dtype=dtype), seed).items()
        }
        for v in params.values():
            if v.dtype.is_floating_point:
                v.requires_grad_(True)

        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        def zero_grads():
            for p in params.values():
                if p.grad is not None:
                    p.grad.zero_()

        # Forward benchmarks
        for method in ["forward_vertices", "forward_skeleton"]:

            def fwd(m=method):
                with torch.no_grad():
                    getattr(model, m)(**params)
                sync()

            r = benchmark_fn(fwd, method, min_run_time)
            r.batch_size = batch_size
            results.append(r)
            print(f"  {method}: {r.mean_ms:.3f} ± {r.std_ms:.3f} ms")

        # Backward benchmarks
        for method in ["forward_vertices", "forward_skeleton"]:
            verify_gradients_flow(model, params, method, device)

            def bwd(m=method):
                getattr(model, m)(**params).sum().backward()
                sync()
                zero_grads()

            backward_name = method.replace("forward_", "backward_")
            r = benchmark_fn(bwd, backward_name, min_run_time)
            r.batch_size = batch_size
            results.append(r)
            print(f"  {backward_name}: {r.mean_ms:.3f} ± {r.std_ms:.3f} ms")

    return results


def load_baseline_lookup(path: Path, baseline_mode: str) -> dict[tuple[str, int], float]:
    data = json.loads(path.read_text())
    results = data.get("results", {}).get(baseline_mode)
    if not results:
        raise ValueError(f"Baseline mode '{baseline_mode}' not found in {path}")
    return {(r["name"], r["batch_size"]): r["mean_ms"] for r in results}


def print_results_table(
    all_results: dict[str, list[BenchmarkResult]],
    batch_sizes: list[int],
    baseline_lookup: dict[tuple[str, int], float] | None = None,
) -> None:
    """Print results as a formatted table."""
    ops = ["forward_vertices", "forward_skeleton", "backward_vertices", "backward_skeleton"]

    print("\n" + "=" * 80)
    if baseline_lookup:
        print("BENCHMARK RESULTS (speedup vs baseline)")
    else:
        print("BENCHMARK RESULTS (all times in ms)")
    print("=" * 80)

    for op in ops:
        print(f"\n{op}:")
        header = f"{'Batch':>6} |" + "".join(f" {mode:>20} |" for mode in all_results)
        print(header)
        print("-" * len(header))

        for bs in batch_sizes:
            row = f"{bs:>6} |"
            for results in all_results.values():
                match = next((r for r in results if r.batch_size == bs and r.name == op), None)
                if not match:
                    row += f" {'N/A':>20} |"
                    continue
                if baseline_lookup:
                    baseline = baseline_lookup.get((op, bs))
                    if baseline:
                        speedup = baseline / match.mean_ms
                        row += f" {speedup:>8.3f}x{'':>8} |"
                    else:
                        row += f" {'N/A':>20} |"
                else:
                    row += f" {match.mean_ms:>8.3f} ± {match.std_ms:>6.3f} |"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark body model performance")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    parser.add_argument("--batch-sizes", type=str, default="1,8,32", help="Comma-separated batch sizes")
    parser.add_argument("--compile", action="store_true", help="Run with torch.compile")
    parser.add_argument("--compile-autotune", action="store_true", help="Run with torch.compile(mode='max-autotune')")
    parser.add_argument(
        "--compile-reduce-overhead", action="store_true", help="Run with torch.compile(mode='reduce-overhead')"
    )
    parser.add_argument("--json-out", type=Path, help="Write benchmark results to a JSON file")
    parser.add_argument("--baseline-json", type=Path, help="Baseline JSON file for speedup comparison output")
    parser.add_argument("--baseline-mode", type=str, default="no_compile", help="Mode in baseline JSON to compare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for model and inputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-run-time", type=float, default=2.0, help="Min benchmark time per op (seconds)")

    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"Benchmarking {args.model} on {args.device} ({args.dtype})")
    print(f"Batch sizes: {batch_sizes}, Seed: {args.seed}")

    all_results: dict[str, list[BenchmarkResult]] = {}

    # Baseline (no compile)
    print("\n--- No compile ---")
    results = run_benchmark(args.model, batch_sizes, None, args.device, dtype, args.seed, args.min_run_time)
    all_results["no_compile"] = results

    # Compile modes
    compile_flags = [
        (args.compile, "default"),
        (args.compile_autotune, "max-autotune"),
        (args.compile_reduce_overhead, "reduce-overhead"),
    ]
    for flag, mode in compile_flags:
        if flag:
            print(f"\n--- torch.compile ({mode}) ---")
            results = run_benchmark(args.model, batch_sizes, mode, args.device, dtype, args.seed, args.min_run_time)
            all_results[mode] = results

    if args.json_out:
        payload = {
            "meta": {
                "model": args.model,
                "device": args.device,
                "dtype": args.dtype,
                "batch_sizes": batch_sizes,
                "seed": args.seed,
                "min_run_time": args.min_run_time,
                "modes": list(all_results.keys()),
            },
            "results": {mode: [asdict(r) for r in results] for mode, results in all_results.items()},
        }
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nWrote JSON results to {args.json_out}")

    baseline_lookup = None
    if args.baseline_json:
        baseline_lookup = load_baseline_lookup(args.baseline_json, args.baseline_mode)
        print(f"\nComparing against baseline: {args.baseline_json} ({args.baseline_mode})")

    print_results_table(all_results, batch_sizes, baseline_lookup)


if __name__ == "__main__":
    main()
