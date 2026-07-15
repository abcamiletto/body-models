"""Benchmark eager and compiled GarmentMeasurements forward/backward passes.

Usage:
    uv run bench/garment_forward_vertices.py --batch-sizes 1,8,32
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from torch import Tensor

from body_models.bodies.garment_measurements.backends import core
from body_models.bodies.smpl.backends import warp as smpl_warp
from body_models.garment_measurements.torch import GarmentMeasurements


def benchmark(function: Callable[[], Tensor], warmup: int, runs: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1_000)
    return statistics.median(timings)


def inference(function: Callable[[], Tensor]) -> Callable[[], Tensor]:
    def call() -> Tensor:
        with torch.inference_mode():
            return function()

    return call


def training_step(function: Callable[[], Tensor], inputs: tuple[Tensor, ...]) -> Callable[[], Tensor]:
    def step() -> Tensor:
        for value in inputs:
            value.grad = None
        vertices = function()
        vertices.square().mean().backward()
        return vertices

    return step


def trainable(params: dict[str, Tensor], *, include_shape: bool) -> dict[str, Tensor]:
    return {
        key: value.detach().clone().requires_grad_() for key, value in params.items() if include_shape or key != "shape"
    }


def dense_skinning(vertices: Tensor, transforms: Tensor, weights: Tensor) -> Tensor:
    return core.linear_blend_skinning(torch, vertices, transforms, weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    models = {kernel: GarmentMeasurements(kernel=kernel).to(device).eval() for kernel in GarmentMeasurements.kernels}
    compiled_skinning = torch.compile(dense_skinning)

    print(f"device={torch.cuda.get_device_name(device)} vertices={models['torch'].num_vertices}")
    for batch_size in map(int, args.batch_sizes.split(",")):
        params = models["torch"].get_rest_pose(batch_dims=(batch_size,))
        identity = models["torch"].prepare_identity(params["shape"])

        print(f"\nB={batch_size} model forward / forward+backward")
        for kernel, model in models.items():

            def full_forward() -> Tensor:
                return model.forward_vertices(**params)

            def prepared_forward() -> Tensor:
                return model.forward_vertices(**params, identity=identity)

            full_params = trainable(params, include_shape=True)

            def full_training() -> Tensor:
                return model.forward_vertices(**full_params)

            full_inputs = tuple(full_params.values())

            prepared_params = trainable(params, include_shape=False)

            def prepared_training() -> Tensor:
                return model.forward_vertices(**prepared_params, identity=identity)

            prepared_inputs = tuple(prepared_params.values())

            full_forward_ms = benchmark(inference(full_forward), args.warmup, args.runs)
            full_backward_ms = benchmark(training_step(full_training, full_inputs), args.warmup, args.runs)
            prepared_forward_ms = benchmark(inference(prepared_forward), args.warmup, args.runs)
            prepared_backward_ms = benchmark(
                training_step(prepared_training, prepared_inputs),
                args.warmup,
                args.runs,
            )
            print(
                f"  {kernel:5s} full={full_forward_ms:8.3f}/{full_backward_ms:8.3f} ms"
                f"  prepared={prepared_forward_ms:8.3f}/{prepared_backward_ms:8.3f} ms"
            )

        pose = models["torch"].prepare_pose(
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
            params["pelvis_rotation"],
            identity=identity,
        )
        rest_vertices = identity["rest_vertices"]
        transforms = pose["skinning_transforms"]
        train_vertices = rest_vertices.detach().requires_grad_()
        train_transforms = transforms.detach().requires_grad_()
        weights = models["torch"].weights
        forward_calls = {
            "torch": lambda: dense_skinning(rest_vertices, transforms, weights.skin_weights),
            "compiled": lambda: compiled_skinning(rest_vertices, transforms, weights.skin_weights),
            "warp": lambda: smpl_warp.warp_affine_blend_skinning(
                rest_vertices,
                transforms,
                weights.skin_joint_indices,
                weights.skin_joint_weights,
            ),
        }
        training_calls = {
            "torch": lambda: dense_skinning(train_vertices, train_transforms, weights.skin_weights),
            "compiled": lambda: compiled_skinning(train_vertices, train_transforms, weights.skin_weights),
            "warp": lambda: smpl_warp.warp_affine_blend_skinning(
                train_vertices,
                train_transforms,
                weights.skin_joint_indices,
                weights.skin_joint_weights,
            ),
        }

        expected = forward_calls["torch"]()
        torch.testing.assert_close(forward_calls["compiled"](), expected)
        torch.testing.assert_close(forward_calls["warp"](), expected, rtol=1e-4, atol=1e-4)
        print("  skinning primitive forward / forward+backward")
        for name in forward_calls:
            forward_ms = benchmark(inference(forward_calls[name]), args.warmup, args.runs)
            backward_ms = benchmark(
                training_step(training_calls[name], (train_vertices, train_transforms)),
                args.warmup,
                args.runs,
            )
            print(f"    {name:8s} {forward_ms:8.3f}/{backward_ms:8.3f} ms")


if __name__ == "__main__":
    main()
