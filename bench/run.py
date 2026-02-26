"""Benchmark forward_skeleton and forward_vertices for all body models.

Usage:
    python bench/run.py                           # Full benchmark suite
    python bench/run.py -m SMPLX                  # Only SMPLX
    python bench/run.py -m SMPLX -m SMPL          # SMPLX and SMPL
    python bench/run.py -d cuda                   # CUDA only
    python bench/run.py -d cuda --compile         # CUDA compiled only
    python bench/run.py --method skeleton          # Skeleton only
    python bench/run.py -m SMPLX -d cuda --compile --method skeleton
    python bench/run.py --batch-sizes 512,1024    # Custom batch sizes
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from body_models.anny.torch import ANNY
from body_models.config import get_model_path
from body_models.flame.torch import FLAME
from body_models.mhr.torch import MHR
from body_models.skel.torch import SKEL
from body_models.smpl.torch import SMPL
from body_models.smplx.torch import SMPLX

DEFAULT_SKELETON_BATCH_SIZES = [256, 512, 1024, 2048, 4096]
DEFAULT_VERTICES_BATCH_SIZES = [64, 128, 256, 512]
DEFAULT_N_RUNS = 20
DEFAULT_WARMUP = 5


def remove_outliers_and_mean(values: list[float], factor: float = 1.5) -> float:
    """Remove outliers using IQR method and return mean of remaining values."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = [v for v in values if lower <= v <= upper]
    return statistics.mean(filtered) if filtered else statistics.mean(values)


ALL_MODEL_NAMES = ["SMPL", "SMPLX", "SKEL", "FLAME", "ANNY", "MHR"]


def load_models(device: torch.device, model_filter: list[str] | None = None) -> dict[str, Any]:
    """Load all available models (or a filtered subset)."""
    wanted = {m.upper() for m in model_filter} if model_filter else set(ALL_MODEL_NAMES)
    models = {}

    # SMPL
    if "SMPL" in wanted:
        path = get_model_path("smpl-neutral")
        if path and path.exists():
            models["SMPL"] = SMPL(model_path=str(path)).to(device).eval()
        else:
            print("  Skipping SMPL (no model path configured)")

    # SMPLX
    if "SMPLX" in wanted:
        path = get_model_path("smplx-neutral")
        if path and path.exists():
            models["SMPLX"] = SMPLX(model_path=str(path)).to(device).eval()
        else:
            print("  Skipping SMPLX (no model path configured)")

    # SKEL
    if "SKEL" in wanted:
        path = get_model_path("skel")
        if path and Path(path).exists():
            models["SKEL"] = SKEL(gender="male", model_path=str(path)).to(device).eval()
        else:
            print("  Skipping SKEL (no model path configured)")

    # FLAME
    if "FLAME" in wanted:
        path = get_model_path("flame")
        if path and Path(path).exists():
            models["FLAME"] = FLAME(model_path=str(path)).to(device).eval()
        else:
            print("  Skipping FLAME (no model path configured)")

    # ANNY (auto-downloads)
    if "ANNY" in wanted:
        try:
            models["ANNY"] = ANNY().to(device).eval()
        except Exception as e:
            print(f"  Skipping ANNY ({e})")

    # MHR (auto-downloads)
    if "MHR" in wanted:
        try:
            models["MHR"] = MHR().to(device).eval()
        except Exception as e:
            print(f"  Skipping MHR ({e})")

    return models


def benchmark_method(
    method,
    params: dict,
    device: torch.device,
    n_runs: int = DEFAULT_N_RUNS,
    warmup: int = DEFAULT_WARMUP,
) -> float:
    """Run a single method benchmark and return mean time in ms."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            method(**params)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            method(**params)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return remove_outliers_and_mean(times)


def benchmark_all(
    device: torch.device,
    compile: bool = False,
    model_filter: list[str] | None = None,
    methods: list[str] | None = None,
    skeleton_batch_sizes: list[int] | None = None,
    vertices_batch_sizes: list[int] | None = None,
    n_runs: int = DEFAULT_N_RUNS,
    warmup: int = DEFAULT_WARMUP,
) -> dict:
    """Benchmark all models and return results dict."""
    label = f"{device}" + (" (compiled)" if compile else "")
    print(f"\nLoading models on {label}...")
    models = load_models(device, model_filter)

    skel_bs = skeleton_batch_sizes or DEFAULT_SKELETON_BATCH_SIZES
    vert_bs = vertices_batch_sizes or DEFAULT_VERTICES_BATCH_SIZES

    method_configs = []
    if methods is None or "skeleton" in methods:
        method_configs.append(("forward_skeleton", skel_bs))
    if methods is None or "vertices" in methods:
        method_configs.append(("forward_vertices", vert_bs))

    results = {}  # {model_name: {(method, batch_size): mean_ms}}

    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        results[name] = {}

        for method_name, batch_sizes in method_configs:
            method = getattr(model, method_name)
            if compile:
                method = torch.compile(method, fullgraph=True)
                # Trigger compilation with a small batch before benchmarking
                warmup_params = model.get_rest_pose(batch_size=2)
                warmup_params = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in warmup_params.items()
                }
                with torch.no_grad():
                    method(**warmup_params)

            for bs in batch_sizes:
                params = model.get_rest_pose(batch_size=bs)
                params = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in params.items()}
                mean_ms = benchmark_method(method, params, device, n_runs=n_runs, warmup=warmup)
                results[name][(method_name, bs)] = mean_ms
                print(f"  {method_name} (B={bs:>4}): {mean_ms:8.2f} ms")

    return results


def format_table(results: dict, method_name: str, batch_sizes: list[int]) -> str:
    """Format results for a single method as a markdown table."""
    # Collect batch sizes that actually appear in the results
    used_bs = [bs for bs in batch_sizes if any((method_name, bs) in r for r in results.values())]
    if not used_bs:
        return ""

    model_names = list(results.keys())
    header = "| Model | " + " | ".join(f"B={bs}" for bs in used_bs) + " |"
    sep = "|---|" + "|".join("---:" for _ in used_bs) + "|"

    rows = []
    for name in model_names:
        cells = []
        for bs in used_bs:
            key = (method_name, bs)
            if key in results[name]:
                cells.append(f"{results[name][key]:.2f}")
            else:
                cells.append("N/A")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join([header, sep, *rows])


def write_markdown(
    all_results: dict[str, dict],
    output_path: Path,
    n_runs: int,
    warmup: int,
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
) -> None:
    """Write benchmark results to a markdown file."""
    lines = [
        "# Benchmark Results",
        "",
        f"- **Runs per measurement**: {n_runs} (outliers removed via IQR)",
        f"- **Warmup runs**: {warmup}",
        f"- **PyTorch version**: {torch.__version__}",
        f"- **CUDA available**: {torch.cuda.is_available()}",
        "",
    ]

    for device_name, results in all_results.items():
        lines.append(f"## {device_name}")
        lines.append("")

        skel_table = format_table(results, "forward_skeleton", skeleton_batch_sizes)
        if skel_table:
            lines.append("### `forward_skeleton` (ms)")
            lines.append("")
            lines.append(skel_table)
            lines.append("")

        vert_table = format_table(results, "forward_vertices", vertices_batch_sizes)
        if vert_table:
            lines.append("### `forward_vertices` (ms)")
            lines.append("")
            lines.append(vert_table)
            lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nResults saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark body models")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help=f"Model(s) to benchmark (can repeat). Choices: {', '.join(ALL_MODEL_NAMES)}",
    )
    parser.add_argument(
        "-d",
        "--device",
        action="append",
        dest="devices",
        metavar="DEV",
        help="Device(s) to benchmark: cpu, cuda (can repeat). Default: all available",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=["skeleton", "vertices"],
        help="Method(s) to benchmark (can repeat). Default: both",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=None,
        help="Only run compiled benchmarks",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Only run eager (non-compiled) benchmarks",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Override batch sizes (comma-separated, e.g. '512,1024,2048'). Applies to both methods.",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help=f"Number of timed runs (default: {DEFAULT_N_RUNS})",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write results to BENCHMARK.md",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve batch sizes
    if args.batch_sizes:
        custom_bs = [int(x.strip()) for x in args.batch_sizes.split(",")]
        skel_bs = custom_bs
        vert_bs = custom_bs
    else:
        skel_bs = DEFAULT_SKELETON_BATCH_SIZES
        vert_bs = DEFAULT_VERTICES_BATCH_SIZES

    # Resolve compile modes
    if args.compile and args.no_compile:
        raise SystemExit("Cannot use both --compile and --no-compile")
    if args.compile:
        compile_modes = [True]
    elif args.no_compile:
        compile_modes = [False]
    else:
        compile_modes = [False, True]

    # Resolve devices
    if args.devices:
        devices = [torch.device(d) for d in args.devices]
    else:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

    output_path = Path(__file__).resolve().parent.parent / "BENCHMARK.md"
    all_results = {}

    for device in devices:
        for do_compile in compile_modes:
            label = f"{device}".upper()
            if do_compile:
                label += " (torch.compile)"
            all_results[label] = benchmark_all(
                device,
                compile=do_compile,
                model_filter=args.models,
                methods=args.methods,
                skeleton_batch_sizes=skel_bs,
                vertices_batch_sizes=vert_bs,
                n_runs=args.runs,
                warmup=args.warmup,
            )

    if not args.no_save:
        write_markdown(all_results, output_path, args.runs, args.warmup, skel_bs, vert_bs)


if __name__ == "__main__":
    main()
