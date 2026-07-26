"""Benchmark skeleton and geometry forwards for public body models.

Examples:
    uv run bench/run.py -m smpl --backend numpy
    uv run bench/run.py -m smpl --backend torch --skinning-backend warp -d cuda
    uv run bench/run.py -m smplx --backend torch -d cuda --prepared-identity
    uv run bench/run.py --method skeleton --batch-sizes 512,1024
    uv run bench/run.py --backend torch --torch-mode eager
"""

from __future__ import annotations

import argparse
import inspect
import platform
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from body_models import catalog, registry
from body_models.runtime import TorchRuntime

DEFAULT_SKELETON_BATCH_SIZES = [256, 512, 1024, 2048, 4096]
DEFAULT_VERTICES_BATCH_SIZES = [64, 128, 256, 512]
DEFAULT_SKELETON_RUNS = 20
DEFAULT_VERTICES_RUNS = 5
DEFAULT_WARMUP = 2
TORCH_COMPILE_MODE = "default"
BACKENDS = ("numpy", "torch")


@dataclass(frozen=True)
class BenchmarkSpec:
    model_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def vertices_method(self) -> str:
        return "forward_links" if catalog.MODEL_SPECS[self.model_name].kind == "rigid" else "forward_vertices"

    @property
    def supports_warp(self) -> bool:
        return catalog.MODEL_SPECS[self.model_name].kind == "skinned"


@dataclass(frozen=True)
class BenchmarkMeasurement:
    first_call_ms: float
    steady_state_ms: float
    peak_memory_mib: float | None


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    measurements: dict[tuple[str, int], BenchmarkMeasurement]


BENCHMARKS = {name: BenchmarkSpec(name) for name in catalog.MODEL_SPECS}
BENCHMARKS |= {
    f"soma-{model_type}": BenchmarkSpec(
        "soma",
        {"model_type": model_type},
    )
    for model_type in ("anny", "mhr", "smpl", "smplx")
}


def main() -> None:
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    skeleton_batch_sizes = batch_sizes or DEFAULT_SKELETON_BATCH_SIZES
    vertices_batch_sizes = batch_sizes or DEFAULT_VERTICES_BATCH_SIZES
    benchmark_names = args.models or list(BENCHMARKS)
    backends = args.backends or list(BACKENDS)
    skinning_backends = args.skinning_backends or list(TorchRuntime.SKINNING_BACKENDS)
    devices = parse_devices(args.devices)
    methods = args.methods or ["skeleton", "vertices"]

    preflight_models(
        benchmark_names,
        backends,
        skinning_backends,
        methods,
        args.prepared_identity,
        args.torch_mode,
    )
    results = benchmark_all(
        benchmark_names=benchmark_names,
        backends=backends,
        skinning_backends=skinning_backends,
        devices=devices,
        methods=methods,
        skeleton_batch_sizes=skeleton_batch_sizes,
        vertices_batch_sizes=vertices_batch_sizes,
        skeleton_runs=args.skeleton_runs,
        vertices_runs=args.vertices_runs,
        warmup=args.warmup,
        prepared_identity=args.prepared_identity,
        torch_mode=args.torch_mode,
    )

    if args.output is not None:
        write_markdown(
            results=results,
            output_path=args.output,
            skeleton_runs=args.skeleton_runs,
            vertices_runs=args.vertices_runs,
            warmup=args.warmup,
            backends=backends,
            skinning_backends=skinning_backends,
            devices=devices,
            methods=methods,
            skeleton_batch_sizes=skeleton_batch_sizes,
            vertices_batch_sizes=vertices_batch_sizes,
            prepared_identity=args.prepared_identity,
            torch_mode=args.torch_mode,
        )


def preflight_models(
    benchmark_names: list[str],
    backends: list[str],
    skinning_backends: list[str],
    methods: list[str],
    prepared_identity: bool,
    torch_mode: str,
) -> None:
    print("Checking model instantiation...")
    for benchmark_name in benchmark_names:
        spec = BENCHMARKS[benchmark_name]
        for backend in backends:
            for skinning_backend in implementations(spec, backend, skinning_backends):
                if not benchmark_methods(methods, skinning_backend):
                    continue
                model = create_model(spec, backend, skinning_backend, torch.device("cpu"))
                uses_prepared_identity = prepared_identity and hasattr(model, "prepare_identity")
                result_label = label(
                    benchmark_name,
                    backend,
                    skinning_backend,
                    prepared_identity=uses_prepared_identity,
                    torch_mode=torch_mode,
                )
                print(f"  {result_label}")


def benchmark_all(
    *,
    benchmark_names: list[str],
    backends: list[str],
    skinning_backends: list[str],
    devices: list[torch.device],
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
    prepared_identity: bool,
    torch_mode: str,
) -> list[BenchmarkResult]:
    results = []
    for benchmark_name in benchmark_names:
        spec = BENCHMARKS[benchmark_name]
        for backend in backends:
            for skinning_backend in implementations(spec, backend, skinning_backends):
                selected_methods = benchmark_methods(methods, skinning_backend)
                if not selected_methods:
                    continue
                backend_devices = devices if backend == "torch" else [None]
                for device in backend_devices:
                    model = create_model(spec, backend, skinning_backend, device)
                    uses_prepared_identity = prepared_identity and hasattr(model, "prepare_identity")
                    result_label = label(
                        benchmark_name,
                        backend,
                        skinning_backend,
                        device,
                        uses_prepared_identity,
                        torch_mode,
                    )
                    results.append(
                        benchmark_model(
                            result_label,
                            model,
                            backend,
                            device,
                            spec,
                            selected_methods,
                            skeleton_batch_sizes,
                            vertices_batch_sizes,
                            skeleton_runs,
                            vertices_runs,
                            warmup,
                            uses_prepared_identity,
                            torch_mode,
                        )
                    )
    return results


def implementations(
    spec: BenchmarkSpec,
    backend: str,
    skinning_backends: list[str],
) -> tuple[str | None, ...]:
    if backend != "torch" or not spec.supports_warp:
        return (None,)
    return tuple(skinning_backends)


def benchmark_methods(methods: list[str], skinning_backend: str | None) -> list[str]:
    if skinning_backend != "warp":
        return methods
    return [method for method in methods if method != "skeleton"]


def create_model(
    spec: BenchmarkSpec,
    backend: str,
    skinning_backend: str | None,
    device: torch.device | None,
) -> Any:
    kwargs = dict(spec.kwargs)
    if skinning_backend is not None:
        kwargs["skinning_backend"] = skinning_backend
    model = registry.create_model(spec.model_name, backend=backend, **kwargs)
    if backend == "torch":
        model = model.to(device).eval()
    return model


def label(
    benchmark_name: str,
    backend: str,
    skinning_backend: str | None,
    device: torch.device | None = None,
    prepared_identity: bool = False,
    torch_mode: str = "compile",
) -> str:
    implementation = backend if skinning_backend is None else f"{backend}/{skinning_backend}"
    if device is not None:
        device_name = "gpu" if device.type == "cuda" else device.type
        implementation = f"{implementation}, {device_name}"
    if backend == "torch":
        execution = "compiled" if torch_mode == "compile" else "eager"
        implementation = f"{implementation}, {execution}"
    if prepared_identity:
        implementation = f"{implementation}, prepared identity"
    return f"{benchmark_name.upper()} ({implementation})"


def benchmark_model(
    result_label: str,
    model: Any,
    backend: str,
    device: torch.device | None,
    spec: BenchmarkSpec,
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
    prepared_identity: bool,
    torch_mode: str,
) -> BenchmarkResult:
    print(f"\nBenchmarking {result_label}...")
    measurements = {}
    configurations = []
    if "skeleton" in methods:
        configurations.append(("forward_skeleton", "forward_skeleton", skeleton_batch_sizes, skeleton_runs))
    if "vertices" in methods:
        configurations.append(("forward_vertices", spec.vertices_method, vertices_batch_sizes, vertices_runs))

    for result_name, method_name, batch_sizes, runs in configurations:
        method = getattr(model, method_name)
        if backend == "torch" and torch_mode == "compile":
            method = torch.compile(method, mode=TORCH_COMPILE_MODE)

        for batch_size in batch_sizes:
            params = benchmark_params(model, batch_size, prepared_identity)
            params = move_tensors(params, device)
            measurement = benchmark_method(method, params, backend, device, runs, warmup)
            measurements[(result_name, batch_size)] = measurement
            memory = f", peak {measurement.peak_memory_mib:.1f} MiB" if measurement.peak_memory_mib is not None else ""
            print(
                f"  {method_name} (B={batch_size:>4}): "
                f"{measurement.steady_state_ms:8.2f} ms steady, "
                f"{measurement.first_call_ms:.2f} ms first{memory}"
            )

    return BenchmarkResult(result_label, measurements)


def benchmark_params(model: Any, batch_size: int, prepared_identity: bool) -> dict[str, Any]:
    params = model.get_rest_pose(batch_dims=(batch_size,))
    if not prepared_identity:
        return params

    identity_params = {}
    for name, parameter in inspect.signature(model.prepare_identity).parameters.items():
        if name in params:
            identity_params[name] = params.pop(name)
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"get_rest_pose() does not provide required identity parameter {name!r}")
    params["identity"] = model.prepare_identity(**identity_params)
    return params


def benchmark_method(
    method: Any,
    params: dict[str, Any],
    backend: str,
    device: torch.device | None,
    runs: int,
    warmup: int,
) -> BenchmarkMeasurement:
    context = torch.inference_mode if backend == "torch" else nullcontext
    synchronize(device)
    start = time.perf_counter()
    with context():
        method(**params)
    synchronize(device)
    first_call_ms = (time.perf_counter() - start) * 1000

    for _ in range(warmup):
        with context():
            method(**params)
        synchronize(device)

    baseline_memory = None
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)

    times = []
    for _ in range(runs):
        synchronize(device)
        start = time.perf_counter()
        with context():
            method(**params)
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)

    peak_memory_mib = None
    if baseline_memory is not None:
        peak_memory = torch.cuda.max_memory_allocated(device) - baseline_memory
        peak_memory_mib = peak_memory / 2**20
    return BenchmarkMeasurement(
        first_call_ms=first_call_ms,
        steady_state_ms=mean_without_outliers(times),
        peak_memory_mib=peak_memory_mib,
    )


def write_markdown(
    *,
    results: list[BenchmarkResult],
    output_path: Path,
    skeleton_runs: int,
    vertices_runs: int,
    warmup: int,
    backends: list[str],
    skinning_backends: list[str],
    devices: list[torch.device],
    methods: list[str],
    skeleton_batch_sizes: list[int],
    vertices_batch_sizes: list[int],
    prepared_identity: bool,
    torch_mode: str,
) -> None:
    torch_devices = ", ".join(device_description(device) for device in devices)
    torch_execution = f"torch.compile(mode={TORCH_COMPILE_MODE!r})" if torch_mode == "compile" else "eager"
    lines = [
        "# Benchmark Results",
        "",
        f"- **Skeleton runs per measurement**: {skeleton_runs} (outliers removed via IQR)",
        f"- **Vertices runs per measurement**: {vertices_runs} (outliers removed via IQR)",
        f"- **Warmup runs**: {warmup}",
        f"- **Backends**: {', '.join(backends)}",
        f"- **Torch skinning backends**: {', '.join(skinning_backends)}",
        f"- **Torch devices**: {torch_devices}",
        f"- **Torch execution**: {torch_execution}",
        f"- **Prepared identities**: {prepared_identity}",
        f"- **Platform**: {platform.platform()}",
        f"- **PyTorch version**: {torch.__version__}",
        f"- **CUDA available**: {torch.cuda.is_available()}",
        "",
    ]
    if "skeleton" in methods:
        lines.extend(
            [
                "## `forward_skeleton` (ms)",
                "",
                format_table(results, "forward_skeleton", skeleton_batch_sizes, "steady_state_ms"),
                "",
                "### First call (ms)",
                "",
                format_table(results, "forward_skeleton", skeleton_batch_sizes, "first_call_ms"),
                "",
                "### Peak CUDA memory (MiB)",
                "",
                format_table(results, "forward_skeleton", skeleton_batch_sizes, "peak_memory_mib"),
                "",
            ]
        )
    if "vertices" in methods:
        lines.extend(
            [
                "## `forward_vertices` / `forward_links` (ms)",
                "",
                format_table(results, "forward_vertices", vertices_batch_sizes, "steady_state_ms"),
                "",
                "### First call (ms)",
                "",
                format_table(results, "forward_vertices", vertices_batch_sizes, "first_call_ms"),
                "",
                "### Peak CUDA memory (MiB)",
                "",
                format_table(results, "forward_vertices", vertices_batch_sizes, "peak_memory_mib"),
                "",
            ]
        )

    output_path.write_text("\n".join(lines))
    print(f"\nResults saved to {output_path}")


def format_table(
    results: list[BenchmarkResult],
    method_name: str,
    batch_sizes: list[int],
    metric: str,
) -> str:
    header = "| Model | " + " | ".join(f"B={batch_size}" for batch_size in batch_sizes) + " |"
    separator = "|---|" + "|".join("---:" for _ in batch_sizes) + "|"
    rows = []
    for result in results:
        measurements = [result.measurements.get((method_name, batch_size)) for batch_size in batch_sizes]
        values = [getattr(measurement, metric) if measurement is not None else None for measurement in measurements]
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


def device_description(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    return f"cuda ({torch.cuda.get_device_name(device)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark body models")
    parser.add_argument(
        "-m",
        "--model",
        action="append",
        dest="models",
        choices=BENCHMARKS,
        help="Model benchmark to run (can repeat). Default: all",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=BACKENDS,
        help="Array backend to benchmark (can repeat). Default: all",
    )
    parser.add_argument(
        "--skinning-backend",
        action="append",
        dest="skinning_backends",
        choices=TorchRuntime.SKINNING_BACKENDS,
        help="Torch skinning backend (can repeat). Default: all",
    )
    parser.add_argument(
        "-d",
        "--device",
        action="append",
        dest="devices",
        metavar="DEV",
        help="Torch device (can repeat). Default: cpu plus cuda when available",
    )
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=["skeleton", "vertices"],
        help="Method to benchmark (can repeat). Default: both",
    )
    parser.add_argument(
        "--batch-sizes",
        help="Override batch sizes with a comma-separated list",
    )
    parser.add_argument("--skeleton-runs", type=int, default=DEFAULT_SKELETON_RUNS)
    parser.add_argument("--vertices-runs", type=int, default=DEFAULT_VERTICES_RUNS)
    parser.add_argument("-w", "--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--prepared-identity",
        action="store_true",
        help="Precompute identity-dependent state for models that support it",
    )
    parser.add_argument(
        "--torch-mode",
        choices=("compile", "eager"),
        default="compile",
        help="Torch execution mode (default: compile)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path",
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


if __name__ == "__main__":
    main()
