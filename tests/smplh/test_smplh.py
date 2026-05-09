"""Tests for the SMPL-H body model."""

from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

from nanomanifold import SO3

pytestmark = pytest.mark.fast

MODEL_PATH = Path(__file__).parents[1] / "assets" / "models_hub" / "smplh-neutral" / "model.npz"
RTOL, ATOL = 1e-4, 1e-4
ROTATION_TYPES = ["axis_angle", "quat", "sixd", "matrix", "rotmat"]

if not MODEL_PATH.exists():
    pytest.skip(f"SMPLH model not found at {MODEL_PATH}", allow_module_level=True)


def _smplh_backend(backend: str):
    if backend == "jax":
        pytest.importorskip("jax.numpy")
    module = import_module(f"body_models.smplh.{backend}")
    return getattr(module, "SMPLH")


def _backend_array(backend: str, value: np.ndarray):
    if backend == "torch":
        return torch.tensor(value)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.array(value)
    return value


def _to_numpy(backend: str, value):
    if backend == "torch":
        return value.detach().numpy()
    return np.asarray(value)


def _rest_pose_inputs(model) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[0] for key, value in model.get_rest_pose(batch_size=1).items()}


def _convert_rotation_inputs(inputs: dict[str, np.ndarray], rotation_type: str) -> dict[str, np.ndarray]:
    if rotation_type == "axis_angle":
        return inputs

    return {
        "shape": inputs["shape"],
        "body_pose": SO3.convert(inputs["body_pose"], src="axis_angle", dst=rotation_type, xp=np),
        "hand_pose": SO3.convert(inputs["hand_pose"], src="axis_angle", dst=rotation_type, xp=np),
        "pelvis_rotation": SO3.convert(inputs["pelvis_rotation"], src="axis_angle", dst=rotation_type, xp=np),
        "global_translation": inputs["global_translation"],
    }


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_forward_rest_pose(backend: str) -> None:
    SMPLH = _smplh_backend(backend)
    model = SMPLH(model_path=MODEL_PATH)
    params = model.get_rest_pose(batch_size=2)

    context = torch.no_grad() if backend == "torch" else nullcontext()
    with context:
        vertices = model.forward_vertices(**params)
        skeleton = model.forward_skeleton(**params)

    vertices_np = _to_numpy(backend, vertices)
    skeleton_np = _to_numpy(backend, skeleton)

    assert vertices_np.shape == (2, 6890, 3)
    assert skeleton_np.shape == (2, 52, 4, 4)
    assert np.isfinite(vertices_np).all()
    assert np.isfinite(skeleton_np).all()


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("rotation_type", ROTATION_TYPES)
def test_rotation_types(rotation_type: str, backend: str) -> None:
    SMPLH = _smplh_backend(backend)
    native_model = SMPLH(model_path=MODEL_PATH)
    rotated_model = SMPLH(model_path=MODEL_PATH, rotation_type=rotation_type)

    inputs = _rest_pose_inputs(native_model)
    inputs["body_pose"] = np.linspace(-0.1, 0.1, native_model.NUM_BODY_JOINTS * 3, dtype=np.float32).reshape(
        native_model.NUM_BODY_JOINTS,
        3,
    )
    inputs["hand_pose"] = np.linspace(-0.05, 0.05, native_model.NUM_HAND_JOINTS * 3, dtype=np.float32).reshape(
        native_model.NUM_HAND_JOINTS,
        3,
    )
    inputs["pelvis_rotation"] = np.array([0.03, -0.02, 0.01], dtype=np.float32)

    rotated_inputs = _convert_rotation_inputs(inputs, rotation_type)
    native_kwargs = {key: _backend_array(backend, value)[None] for key, value in inputs.items()}
    rotated_kwargs = {key: _backend_array(backend, value)[None] for key, value in rotated_inputs.items()}

    context = torch.no_grad() if backend == "torch" else nullcontext()
    with context:
        native_vertices = native_model.forward_vertices(**native_kwargs)
        native_skeleton = native_model.forward_skeleton(**native_kwargs)
        rotated_vertices = rotated_model.forward_vertices(**rotated_kwargs)
        rotated_skeleton = rotated_model.forward_skeleton(**rotated_kwargs)

    np.testing.assert_allclose(
        _to_numpy(backend, rotated_vertices),
        _to_numpy(backend, native_vertices),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        _to_numpy(backend, rotated_skeleton),
        _to_numpy(backend, native_skeleton),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_vertex_subset_matches_full_output(backend: str) -> None:
    SMPLH = _smplh_backend(backend)
    model = SMPLH(model_path=MODEL_PATH)
    kwargs = {key: _backend_array(backend, value)[None] for key, value in _rest_pose_inputs(model).items()}
    vertex_indices = [0, 10, 1, 10, 25]

    context = torch.no_grad() if backend == "torch" else nullcontext()
    with context:
        vertices_full = model.forward_vertices(**kwargs)
        vertices_subset = model.forward_vertices(**kwargs, vertex_indices=vertex_indices)

    np.testing.assert_allclose(
        _to_numpy(backend, vertices_subset),
        _to_numpy(backend, vertices_full)[:, vertex_indices],
        rtol=RTOL,
        atol=ATOL,
    )


def test_torch_warp_kernel_matches_torch() -> None:
    pytest.importorskip("warp")
    from body_models.smplh.torch import SMPLH

    model = SMPLH(model_path=MODEL_PATH, kernel="warp")
    reference_model = SMPLH(model_path=MODEL_PATH)
    params = reference_model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        vertices = model.forward_vertices(**params)
        reference = reference_model.forward_vertices(**params)
        subset = model.forward_vertices(**params, vertex_indices=[0, 10, 1, 10, 25])

    np.testing.assert_allclose(vertices.numpy(), reference.numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(subset.numpy(), reference[:, [0, 10, 1, 10, 25]].numpy(), rtol=RTOL, atol=ATOL)


def test_simplify() -> None:
    from body_models.smplh.torch import SMPLH

    model_orig = SMPLH(model_path=MODEL_PATH, simplify=1.0)
    model_2x = SMPLH(model_path=MODEL_PATH, simplify=2.0)

    assert model_2x.num_vertices < model_orig.num_vertices
    assert model_2x.faces.shape[0] < model_orig.faces.shape[0]

    params = model_2x.get_rest_pose(batch_size=2)
    with torch.no_grad():
        vertices = model_2x.forward_vertices(**params)
        skeleton = model_2x.forward_skeleton(**params)

    assert vertices.shape == (2, model_2x.num_vertices, 3)
    assert skeleton.shape == (2, 52, 4, 4)
