"""Benchmark forward_skeleton and geometry forwards for supported body models.

Usage:
    uv run bench/run.py
    uv run bench/run.py -m SMPLX
    uv run bench/run.py -m SMPLX -m SMPL
    uv run bench/run.py --backend numpy
    uv run bench/run.py -m SMPL --backend numpy --kernel numba
    uv run bench/run.py -m SMPL --backend torch --kernel warp -d cuda
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
BACKENDS = ("numpy", "torch")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    numpy: Callable[[str], Any] | None
    torch: Callable[[str, torch.device], torch.nn.Module] | None
    numpy_kernels: tuple[str, ...] = ("numpy",)
    torch_kernels: tuple[str, ...] = ("torch",)
    prepare_identity: bool = False
    vertices_method: str = "forward_vertices"


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    timings: dict[tuple[str, int], float]


def torch_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return model.to(device).eval()


MODELS = [
    ModelSpec(
        "SMPL",
        lambda kernel: smpl_numpy.SMPL(gender="neutral", kernel=kernel),
        lambda kernel, d: torch_model(smpl_torch.SMPL(gender="neutral", kernel=kernel), d),
        numpy_kernels=smpl_numpy.SMPL.kernels,
        torch_kernels=smpl_torch.SMPL.kernels,
    ),
    ModelSpec(
        "SMPLH",
        lambda kernel: smplh_numpy.SMPLH(gender="neutral", kernel=kernel),
        lambda kernel, d: torch_model(smplh_torch.SMPLH(gender="neutral", kernel=kernel), d),
        numpy_kernels=smplh_numpy.SMPLH.kernels,
        torch_kernels=smplh_torch.SMPLH.kernels,
    ),
    ModelSpec(
        "SMPLX",
        lambda kernel: smplx_numpy.SMPLX(gender="neutral", kernel=kernel),
        lambda kernel, d: torch_model(smplx_torch.SMPLX(gender="neutral", kernel=kernel), d),
        numpy_kernels=smplx_numpy.SMPLX.kernels,
        torch_kernels=smplx_torch.SMPLX.kernels,
    ),
    ModelSpec(
        "MANO",
        lambda kernel: mano_numpy.MANO(side="left", kernel=kernel),
        lambda kernel, d: torch_model(mano_torch.MANO(side="left", kernel=kernel), d),
        numpy_kernels=mano_numpy.MANO.kernels,
        torch_kernels=mano_torch.MANO.kernels,
    ),
    ModelSpec(
        "SKEL",
        lambda _kernel: skel_numpy.SKEL(gender="male"),
        lambda _kernel, d: torch_model(skel_torch.SKEL(gender="male"), d),
    ),
    ModelSpec(
        "FLAME",
        lambda kernel: flame_numpy.FLAME(kernel=kernel),
        lambda kernel, d: torch_model(flame_torch.FLAME(kernel=kernel), d),
        numpy_kernels=flame_numpy.FLAME.kernels,
        torch_kernels=flame_torch.FLAME.kernels,
    ),
    ModelSpec(
        "ANNY",
        lambda kernel: anny_numpy.ANNY(kernel=kernel),
        lambda kernel, d: torch_model(anny_torch.ANNY(kernel=kernel), d),
        numpy_kernels=anny_numpy.ANNY.kernels,
        torch_kernels=anny_torch.ANNY.kernels,
    ),
    ModelSpec("MHR", lambda _kernel: mhr_numpy.MHR(), lambda _kernel, d: torch_model(mhr_torch.MHR(), d)),
    ModelSpec(
        "BRAINCO",
        lambda _kernel: brainco_numpy.BrainCoHand(side="right"),
        None,
        vertices_method="forward_links",
    ),
    ModelSpec(
        "G1",
        lambda _kernel: g1_numpy.G1(),
        lambda _kernel, d: torch_model(g1_torch.G1(), d),
        vertices_method="forward_links",
    ),
    ModelSpec(
        "SOMA",
        lambda kernel: soma_numpy.SOMA(model_type="soma", kernel=kernel),
        lambda kernel, d: torch_model(soma_torch.SOMA(model_type="soma", kernel=kernel), d),
        numpy_kernels=soma_numpy.SOMA.kernels,
        torch_kernels=soma_torch.SOMA.kernels,
        prepare_identity=True,
    ),
    ModelSpec(
        "SOMA-ANNY",
        lambda kernel: soma_numpy.SOMA(model_type="anny", kernel=kernel),
        lambda kernel, d: torch_model(soma_torch.SOMA(model_type="anny", kernel=kernel), d),
        numpy_kernels=soma_numpy.SOMA.kernels,
        torch_kernels=soma_torch.SOMA.kernels,
        prepare_identity=True,
    ),
    ModelSpec(
        "SOMA-MHR",
        lambda kernel: soma_numpy.SOMA(model_type="mhr", kernel=kernel),
        lambda kernel, d: torch_model(soma_torch.SOMA(model_type="mhr", kernel=kernel), d),
        numpy_kernels=soma_numpy.SOMA.kernels,
        torch_kernels=soma_torch.SOMA.kernels,
        prepare_identity=True,
    ),
    ModelSpec(
        "SOMA-SMPL",
        lambda kernel: soma_numpy.SOMA(model_type="smpl", kernel=kernel),
        lambda kernel, d: torch_model(soma_torch.SOMA(model_type="smpl", kernel=kernel), d),
        numpy_kernels=soma_numpy.SOMA.kernels,
        torch_kernels=soma_torch.SOMA.kernels,
        prepare_identity=True,
    ),
    ModelSpec(
        "SOMA-SMPLX",
        lambda kernel: soma_numpy.SOMA(model_type="smplx", kernel=kernel),
        lambda kernel, d: torch_model(soma_torch.SOMA(model_type="smplx", kernel=kernel), d),
        numpy_kernels=soma_numpy.SOMA.kernels,
        torch_kernels=soma_torch.SOMA.kernels,
        prepare_identity=True,
    ),
    ModelSpec(
        "GARMENT-MEASUREMENTS",
        lambda kernel: garment_measurements_numpy.GarmentMeasurements(kernel=kernel),
        lambda _kernel, d: torch_model(garment_measurements_torch.GarmentMeasurements(), d),
        numpy_kernels=garment_measurements_numpy.GarmentMeasurements.kernels,
    ),
    ModelSpec(
        "MYOFULLBODY",
        lambda _kernel: myofullbody_numpy.MyoFullBody(),
        lambda _kernel, d: torch_model(myofullbody_torch.MyoFullBody(), d),
        vertices_method="forward_links",
    ),
]
MODEL_NAMES = [model.name for model in MODELS]


def main() -> None:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    skeleton_batch_sizes = batch_sizes or DEFAULT_SKELETON_BATCH_SIZES
    vertices_batch_sizes = batch_sizes or DEFAULT_VERTICES_BATCH_SIZES
    backends = args.backends or list(BACKENDS)
    kernels = args.kernels or list(model_kernels())
    devices = parse_devices(args.devices)
    methods = args.methods or ["skeleton", "vertices"]
    preflight_models(args.models or MODEL_NAMES, backends, kernels)
    results = benchmark_all(
        model_names=args.models or MODEL_NAMES,
        backends=backends,
        kernels=kernels,
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
        kernels=kernels,
        devices=devices,
        methods=methods,
        skeleton_batch_sizes=skeleton_batch_sizes,
        vertices_batch_sizes=vertices_batch_sizes,
    )


def preflight_models(model_names: list[str], backends: list[str], kernels: list[str]) -> None:
    print("Checking model instantiation...")
    wanted = {normalize_model_name(name) for name in model_names}

    for spec in MODELS:
        if spec.name not in wanted:
            continue

        if "numpy" in backends and spec.numpy is not None:
            for kernel in spec.numpy_kernels:
                if kernel not in kernels:
                    continue
                spec.numpy(kernel)
                print(f"  {spec.name} (numpy/{kernel})")

        if "torch" in backends and spec.torch is not None:
            for kernel in spec.torch_kernels:
                if kernel not in kernels:
                    continue
                spec.torch(kernel, torch.device("cpu"))
                print(f"  {spec.name} (torch/{kernel})")


def model_kernels() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            kernel
            for model in MODELS
            for model_kernels in (model.numpy_kernels, model.torch_kernels)
            for kernel in model_kernels
        )
    )


def benchmark_all(
    *,
    model_names: list[str],
    backends: list[str],
    kernels: list[str],
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
            for kernel in spec.numpy_kernels:
                if kernel not in kernels:
                    continue
                label = f"{spec.name} (numpy/{kernel})"
                result = benchmark_model(
                    label,
                    spec.numpy(kernel),
                    "numpy",
                    None,
                    spec.prepare_identity,
                    spec.vertices_method,
                    methods,
                    skeleton_batch_sizes,
                    vertices_batch_sizes,
                    skeleton_runs,
                    vertices_runs,
                    warmup,
                )
                results.append(result)

        if "torch" in backends and spec.torch is not None:
            for kernel in spec.torch_kernels:
                if kernel not in kernels:
                    continue
                for device in devices:
                    device_name = "gpu" if device.type == "cuda" else device.type
                    label = f"{spec.name} (torch/{kernel}, {device_name})"
                    result = benchmark_model(
                        label,
                        spec.torch(kernel, device),
                        "torch",
                        device,
                        spec.prepare_identity,
                        spec.vertices_method,
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
    prepare_identity: bool,
    vertices_method: str,
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
        method_configs.append(("forward_skeleton", "forward_skeleton", skeleton_batch_sizes, skeleton_runs))
    if "vertices" in methods:
        method_configs.append(("forward_vertices", vertices_method, vertices_batch_sizes, vertices_runs))

    for result_name, method_name, batch_sizes, runs in method_configs:
        method = getattr(model, method_name)
        if backend == "torch":
            method = compile_method(method, model, device, prepare_identity)

        for batch_size in batch_sizes:
            params = benchmark_params(model, batch_size, prepare_identity)
            params = move_tensors(params, device)
            mean_ms = benchmark_method(method, params, backend, device, runs, warmup)
            results[(result_name, batch_size)] = mean_ms
            print(f"  {method_name} (B={batch_size:>4}): {mean_ms:8.2f} ms")

    return BenchmarkResult(label, results)


def benchmark_params(model: Any, batch_size: int, prepare_identity: bool = False) -> dict[str, Any]:
    params = model.get_rest_pose(batch_dims=(batch_size,))
    if not prepare_identity:
        return params

    identity = params.pop("identity", None)
    scale_params = params.pop("scale_params", None)
    prepared_identity = model.prepare_identity(identity=identity, scale_params=scale_params, pose=params["pose"])
    params["prepared_identity"] = prepared_identity
    return params


def compile_method(method: Any, model: Any, device: torch.device | None, prepare_identity: bool) -> Any:
    method = torch.compile(method, mode=TORCH_COMPILE_MODE)
    params = benchmark_params(model, batch_size=2, prepare_identity=prepare_identity)
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
    kernels: list[str],
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
        f"- **Kernels**: {', '.join(kernels)}",
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
        lines.extend(["## `forward_vertices` / `forward_links` (ms)", "", vertices_table, ""])

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
    kernels = model_kernels()
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
        help=f"Backend(s) to benchmark: {', '.join(BACKENDS)} (can repeat). Default: all",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        dest="kernels",
        choices=kernels,
        help=f"Kernel(s) to benchmark: {', '.join(kernels)} (can repeat). Default: all",
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
