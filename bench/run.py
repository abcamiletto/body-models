"""Benchmark forward_skeleton and forward_vertices for supported body models.

Usage:
    uv run bench/run.py
    uv run bench/run.py -m SMPLX
    uv run bench/run.py -m SMPLX -m SMPL
    uv run bench/run.py --backend numpy
    uv run bench/run.py --backend torch -d cuda
    uv run bench/run.py --method skeleton
    uv run bench/run.py --batch-sizes 512,1024
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from body_models.anny import numpy as anny_numpy
from body_models.anny import torch as anny_torch
from body_models.brainco import numpy as brainco_numpy
from body_models.config import get_model_path
from body_models.flame import numpy as flame_numpy
from body_models.flame import torch as flame_torch
from body_models.g1 import numpy as g1_numpy
from body_models.g1 import torch as g1_torch
from body_models.garment_measurements import numpy as garment_measurements_numpy
from body_models.garment_measurements import torch as garment_measurements_torch
from body_models.mano import numpy as mano_numpy
from body_models.mano import torch as mano_torch
from body_models.mhr import numpy as mhr_numpy
from body_models.mhr import torch as mhr_torch
from body_models.myofullbody import numpy as myofullbody_numpy
from body_models.myofullbody import torch as myofullbody_torch
from body_models.skel import numpy as skel_numpy
from body_models.skel import torch as skel_torch
from body_models.smpl import numpy as smpl_numpy
from body_models.smpl import torch as smpl_torch
from body_models.smplh import numpy as smplh_numpy
from body_models.smplh import torch as smplh_torch
from body_models.smplx import numpy as smplx_numpy
from body_models.smplx import torch as smplx_torch
from body_models.soma import numpy as soma_numpy
from body_models.soma import torch as soma_torch

DEFAULT_SKELETON_BATCH_SIZES = [256, 512, 1024, 2048, 4096]
DEFAULT_VERTICES_BATCH_SIZES = [64, 128, 256, 512]
DEFAULT_SKELETON_RUNS = 20
DEFAULT_VERTICES_RUNS = 5
DEFAULT_WARMUP = 2
TORCH_COMPILE_MODE = "default"
BACKENDS = ["numpy", "torch"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    numpy: Callable[[], Any] | None
    torch: Callable[[torch.device], torch.nn.Module] | None


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    timings: dict[tuple[str, int], float]


def path(model: str) -> str:
    model_path = get_model_path(model)
    if model_path is None:
        raise FileNotFoundError(f"No configured path for {model}")
    return str(model_path)


def torch_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return model.to(device).eval()


MODELS = [
    ModelSpec(
        "SMPL",
        lambda: smpl_numpy.SMPL(model_path=path("smpl-neutral")),
        lambda d: torch_model(smpl_torch.SMPL(model_path=path("smpl-neutral")), d),
    ),
    ModelSpec(
        "SMPLH",
        lambda: smplh_numpy.SMPLH(model_path=path("smplh-neutral")),
        lambda d: torch_model(smplh_torch.SMPLH(model_path=path("smplh-neutral")), d),
    ),
    ModelSpec(
        "SMPLX",
        lambda: smplx_numpy.SMPLX(model_path=path("smplx-neutral")),
        lambda d: torch_model(smplx_torch.SMPLX(model_path=path("smplx-neutral")), d),
    ),
    ModelSpec(
        "MANO",
        lambda: mano_numpy.MANO(model_path=path("mano-right")),
        lambda d: torch_model(mano_torch.MANO(model_path=path("mano-right")), d),
    ),
    ModelSpec(
        "SKEL",
        lambda: skel_numpy.SKEL(model_path=path("skel"), gender="male"),
        lambda d: torch_model(skel_torch.SKEL(model_path=path("skel"), gender="male"), d),
    ),
    ModelSpec(
        "FLAME",
        lambda: flame_numpy.FLAME(model_path=path("flame")),
        lambda d: torch_model(flame_torch.FLAME(model_path=path("flame")), d),
    ),
    ModelSpec("ANNY", lambda: anny_numpy.ANNY(), lambda d: torch_model(anny_torch.ANNY(), d)),
    ModelSpec("MHR", lambda: mhr_numpy.MHR(), lambda d: torch_model(mhr_torch.MHR(), d)),
    ModelSpec("BRAINCO", lambda: brainco_numpy.BrainCoHand(), None),
    ModelSpec("G1", lambda: g1_numpy.G1(), lambda d: torch_model(g1_torch.G1(), d)),
    ModelSpec(
        "SOMA",
        lambda: soma_numpy.SOMA(model_type="soma"),
        lambda d: torch_model(soma_torch.SOMA(model_type="soma"), d),
    ),
    ModelSpec(
        "SOMA-ANNY",
        lambda: soma_numpy.SOMA(model_path=path("soma"), model_type="anny"),
        lambda d: torch_model(soma_torch.SOMA(model_path=path("soma"), model_type="anny"), d),
    ),
    ModelSpec(
        "SOMA-MHR",
        lambda: soma_numpy.SOMA(model_path=path("soma"), model_type="mhr"),
        lambda d: torch_model(soma_torch.SOMA(model_path=path("soma"), model_type="mhr"), d),
    ),
    ModelSpec(
        "SOMA-SMPL",
        lambda: soma_numpy.SOMA(model_path=path("soma"), model_type="smpl"),
        lambda d: torch_model(soma_torch.SOMA(model_path=path("soma"), model_type="smpl"), d),
    ),
    ModelSpec(
        "SOMA-SMPLX",
        lambda: soma_numpy.SOMA(model_path=path("soma"), model_type="smplx"),
        lambda d: torch_model(soma_torch.SOMA(model_path=path("soma"), model_type="smplx"), d),
    ),
    ModelSpec(
        "GARMENT-MEASUREMENTS",
        lambda: garment_measurements_numpy.GarmentMeasurements(model_path=path("garment-measurements")),
        lambda d: torch_model(
            garment_measurements_torch.GarmentMeasurements(model_path=path("garment-measurements")), d
        ),
    ),
    ModelSpec(
        "MYOFULLBODY",
        lambda: myofullbody_numpy.MyoFullBody(model_path=path("myofullbody")),
        lambda d: torch_model(myofullbody_torch.MyoFullBody(model_path=path("myofullbody")), d),
    ),
]
MODEL_NAMES = [model.name for model in MODELS]


def main() -> None:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    skeleton_batch_sizes = batch_sizes or DEFAULT_SKELETON_BATCH_SIZES
    vertices_batch_sizes = batch_sizes or DEFAULT_VERTICES_BATCH_SIZES
    backends = args.backends or BACKENDS
    devices = parse_devices(args.devices)
    methods = args.methods or ["skeleton", "vertices"]
    results = benchmark_all(
        model_names=args.models or MODEL_NAMES,
        backends=backends,
        devices=devices,
        methods=methods,
        skeleton_batch_sizes=skeleton_batch_sizes,
        vertices_batch_sizes=vertices_batch_sizes,
        skeleton_runs=args.skeleton_runs,
        vertices_runs=args.vertices_runs,
        warmup=args.warmup,
    )

    if args.no_save:
        return

    output_path = Path(__file__).resolve().parent.parent / "BENCHMARK.md"
    write_markdown(
        results=results,
        output_path=output_path,
        skeleton_runs=args.skeleton_runs,
        vertices_runs=args.vertices_runs,
        warmup=args.warmup,
        backends=backends,
        devices=devices,
        methods=methods,
        skeleton_batch_sizes=skeleton_batch_sizes,
        vertices_batch_sizes=vertices_batch_sizes,
    )


def benchmark_all(
    *,
    model_names: list[str],
    backends: list[str],
    devices: list[torch.device],
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
) -> list[BenchmarkResult]:
    results = []
    wanted = {normalize_model_name(name) for name in model_names}

    for spec in MODELS:
        if spec.name not in wanted:
            continue

        if "numpy" in backends and spec.numpy is not None:
            label = f"{spec.name} (numpy)"
            result = benchmark_model(
                label,
                spec.numpy(),
                "numpy",
                None,
                methods,
                skeleton_batch_sizes,
                vertices_batch_sizes,
                skeleton_runs,
                vertices_runs,
                warmup,
            )
            results.append(result)

        if "torch" in backends and spec.torch is not None:
            for device in devices:
                device_name = "gpu" if device.type == "cuda" else device.type
                label = f"{spec.name} (torch, {device_name})"
                result = benchmark_model(
                    label,
                    spec.torch(device),
                    "torch",
                    device,
                    methods,
                    skeleton_batch_sizes,
                    vertices_batch_sizes,
                    skeleton_runs,
                    vertices_runs,
                    warmup,
                )
                results.append(result)

    return results


def benchmark_model(
    label: str,
    model: Any,
    backend: str,
    device: torch.device | None,
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
) -> BenchmarkResult:
    print(f"\nBenchmarking {label}...")
    results = {}
    method_configs = []
    if "skeleton" in methods:
        method_configs.append(("forward_skeleton", skeleton_batch_sizes, skeleton_runs))
    if "vertices" in methods:
        method_configs.append(("forward_vertices", vertices_batch_sizes, vertices_runs))

    for method_name, batch_sizes, runs in method_configs:
        method = getattr(model, method_name)
        if backend == "torch":
            method = compile_method(method, model, device)

        for batch_size in batch_sizes:
            params = model.get_rest_pose(batch_size=batch_size)
            params = move_tensors(params, device)
            mean_ms = benchmark_method(method, params, backend, device, runs, warmup)
            results[(method_name, batch_size)] = mean_ms
            print(f"  {method_name} (B={batch_size:>4}): {mean_ms:8.2f} ms")

    return BenchmarkResult(label, results)


def compile_method(method, model: torch.nn.Module, device: torch.device):
    method = torch.compile(method, mode=TORCH_COMPILE_MODE)
    params = model.get_rest_pose(batch_size=2)
    params = move_tensors(params, device)
    with torch.inference_mode():
        method(**params)
    return method


def benchmark_method(
    method,
    params: dict[str, Any],
    backend: str,
    device: torch.device | None,
    n_runs: int,
    warmup: int,
) -> float:
    context = torch.inference_mode if backend == "torch" else nullcontext

    for _ in range(warmup):
        with context():
            method(**params)
        synchronize(device)

    times = []
    for _ in range(n_runs):
        synchronize(device)
        start = time.perf_counter()
        with context():
            method(**params)
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)

    return mean_without_outliers(times)


def write_markdown(
    *,
    results: list[BenchmarkResult],
    output_path: Path,
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
    backends: list[str],
    devices: list[torch.device],
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
) -> None:
    torch_devices = ", ".join("gpu" if device.type == "cuda" else device.type for device in devices)
    lines = [
        "# Benchmark Results",
        "",
        f"- **Skeleton runs per measurement**: {skeleton_runs} (outliers removed via IQR)",
        f"- **Vertices runs per measurement**: {vertices_runs} (outliers removed via IQR)",
        f"- **Warmup runs**: {warmup}",
        f"- **Backends**: {', '.join(backends)}",
        f"- **Torch devices**: {torch_devices}",
        f"- **Torch mode**: `torch.compile(mode={TORCH_COMPILE_MODE!r})`",
        f"- **PyTorch version**: {torch.__version__}",
        f"- **CUDA available**: {torch.cuda.is_available()}",
        "",
    ]

    if "skeleton" in methods:
        skeleton_table = format_table(results, "forward_skeleton", skeleton_batch_sizes)
        lines.extend(["## `forward_skeleton` (ms)", "", skeleton_table, ""])
    if "vertices" in methods:
        vertices_table = format_table(results, "forward_vertices", vertices_batch_sizes)
        lines.extend(["## `forward_vertices` (ms)", "", vertices_table, ""])

    output_path.write_text("\n".join(lines))
    print(f"\nResults saved to {output_path}")


def format_table(results: list[BenchmarkResult], method_name: str, batch_sizes: list[int]) -> str:
    header = "| Model | " + " | ".join(f"B={batch_size}" for batch_size in batch_sizes) + " |"
    separator = "|---|" + "|".join("---:" for _ in batch_sizes) + "|"
    rows = []

    for result in results:
        values = [result.timings.get((method_name, batch_size)) for batch_size in batch_sizes]
        cells = [f"{value:.2f}" if value is not None else "N/A" for value in values]
        rows.append(f"| {result.label} | " + " | ".join(cells) + " |")

    return "\n".join([header, separator, *rows])


def mean_without_outliers(values: list[float]) -> float:
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [value for value in values if lower <= value <= upper]
    return statistics.mean(filtered or values)


def move_tensors(params: dict[str, Any], device: torch.device | None) -> dict[str, Any]:
    if device is None:
        return params
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in params.items()}


def synchronize(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark body models")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        metavar="NAME",
        help=f"Model(s) to benchmark (can repeat). Choices: {', '.join(MODEL_NAMES)}",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=BACKENDS,
        help="Backend(s) to benchmark: numpy, torch (can repeat). Default: both",
    )
    parser.add_argument(
        "-d",
        "--device",
        action="append",
        dest="devices",
        metavar="DEV",
        help="Torch device(s) to benchmark: cpu, cuda (can repeat). Default: cpu plus cuda when available",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=["skeleton", "vertices"],
        help="Method(s) to benchmark (can repeat). Default: both",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        help="Override batch sizes (comma-separated, e.g. '512,1024,2048'). Applies to both methods.",
    )
    parser.add_argument(
        "--skeleton-runs",
        type=int,
        default=DEFAULT_SKELETON_RUNS,
        help=f"Timed runs for forward_skeleton (default: {DEFAULT_SKELETON_RUNS})",
    )
    parser.add_argument(
        "--vertices-runs",
        type=int,
        default=DEFAULT_VERTICES_RUNS,
        help=f"Timed runs for forward_vertices (default: {DEFAULT_VERTICES_RUNS})",
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


def parse_batch_sizes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",")]


def parse_devices(values: list[str] | None) -> list[torch.device]:
    if values is not None:
        return [torch.device(value) for value in values]

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def normalize_model_name(name: str) -> str:
    return name.upper().replace("_", "-")


if __name__ == "__main__":
    main()
