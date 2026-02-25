"""Benchmark forward_skeleton and forward_vertices for all body models."""

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

SKELETON_BATCH_SIZES = [256, 512, 1024, 2048, 4096]
VERTICES_BATCH_SIZES = [64, 128, 256, 512]
N_RUNS = 20
WARMUP = 5


def remove_outliers_and_mean(values: list[float], factor: float = 1.5) -> float:
    """Remove outliers using IQR method and return mean of remaining values."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = [v for v in values if lower <= v <= upper]
    return statistics.mean(filtered) if filtered else statistics.mean(values)


def load_models(device: torch.device) -> dict[str, Any]:
    """Load all available models."""
    models = {}

    # SMPL
    path = get_model_path("smpl-neutral")
    if path and path.exists():
        models["SMPL"] = SMPL(model_path=str(path)).to(device).eval()
    else:
        print("  Skipping SMPL (no model path configured)")

    # SMPLX
    path = get_model_path("smplx-neutral")
    if path and path.exists():
        models["SMPLX"] = SMPLX(model_path=str(path)).to(device).eval()
    else:
        print("  Skipping SMPLX (no model path configured)")

    # SKEL
    path = get_model_path("skel")
    if path and Path(path).exists():
        models["SKEL"] = SKEL(gender="male", model_path=str(path)).to(device).eval()
    else:
        print("  Skipping SKEL (no model path configured)")

    # FLAME
    path = get_model_path("flame")
    if path and Path(path).exists():
        models["FLAME"] = FLAME(model_path=str(path)).to(device).eval()
    else:
        print("  Skipping FLAME (no model path configured)")

    # ANNY (auto-downloads)
    try:
        models["ANNY"] = ANNY().to(device).eval()
    except Exception as e:
        print(f"  Skipping ANNY ({e})")

    # MHR (auto-downloads)
    try:
        models["MHR"] = MHR().to(device).eval()
    except Exception as e:
        print(f"  Skipping MHR ({e})")

    return models


def benchmark_method(
    method,
    params: dict,
    device: torch.device,
) -> float:
    """Run a single method benchmark and return mean time in ms."""
    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            method(**params)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(N_RUNS):
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


def benchmark_all(device: torch.device, compile: bool = False) -> dict:
    """Benchmark all models and return results dict."""
    label = f"{device}" + (" (compiled)" if compile else "")
    print(f"\nLoading models on {label}...")
    models = load_models(device)

    results = {}  # {model_name: {(method, batch_size): mean_ms}}

    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        results[name] = {}

        for method_name, batch_sizes in [
            ("forward_skeleton", SKELETON_BATCH_SIZES),
            ("forward_vertices", VERTICES_BATCH_SIZES),
        ]:
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
                mean_ms = benchmark_method(method, params, device)
                results[name][(method_name, bs)] = mean_ms
                print(f"  {method_name} (B={bs:>4}): {mean_ms:8.2f} ms")

    return results


def format_table(results: dict, method_name: str, batch_sizes: list[int]) -> str:
    """Format results for a single method as a markdown table."""
    model_names = list(results.keys())
    header = "| Model | " + " | ".join(f"B={bs}" for bs in batch_sizes) + " |"
    sep = "|---|" + "|".join("---:" for _ in batch_sizes) + "|"

    rows = []
    for name in model_names:
        cells = []
        for bs in batch_sizes:
            key = (method_name, bs)
            if key in results[name]:
                cells.append(f"{results[name][key]:.2f}")
            else:
                cells.append("N/A")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join([header, sep, *rows])


def write_markdown(all_results: dict[str, dict], output_path: Path) -> None:
    """Write benchmark results to a markdown file."""
    lines = [
        "# Benchmark Results",
        "",
        f"- **Runs per measurement**: {N_RUNS} (outliers removed via IQR)",
        f"- **Warmup runs**: {WARMUP}",
        f"- **PyTorch version**: {torch.__version__}",
        f"- **CUDA available**: {torch.cuda.is_available()}",
        "",
    ]

    for device_name, results in all_results.items():
        lines.append(f"## {device_name}")
        lines.append("")
        lines.append("### `forward_skeleton` (ms)")
        lines.append("")
        lines.append(format_table(results, "forward_skeleton", SKELETON_BATCH_SIZES))
        lines.append("")
        lines.append("### `forward_vertices` (ms)")
        lines.append("")
        lines.append(format_table(results, "forward_vertices", VERTICES_BATCH_SIZES))
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nResults saved to {output_path}")


def main():
    output_path = Path(__file__).resolve().parent.parent / "BENCHMARK.md"

    all_results = {}

    # CPU benchmark
    all_results["CPU"] = benchmark_all(torch.device("cpu"))

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        all_results["CUDA"] = benchmark_all(torch.device("cuda"))

    # torch.compile benchmarks
    all_results["CPU (torch.compile)"] = benchmark_all(torch.device("cpu"), compile=True)
    if torch.cuda.is_available():
        all_results["CUDA (torch.compile)"] = benchmark_all(torch.device("cuda"), compile=True)

    write_markdown(all_results, output_path)


if __name__ == "__main__":
    main()
