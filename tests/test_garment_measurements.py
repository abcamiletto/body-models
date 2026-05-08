"""Tests for the GarmentMeasurements PCA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

from body_models.garment_measurements import io
from gradient_utils import prepare_params, sampled_gradcheck

pytestmark = pytest.mark.fast

ASSET_DIR = Path(__file__).parent / "assets" / "models_hub" / "garment_measurements"
MODEL_PATH = ASSET_DIR
MODEL_FILE = MODEL_PATH / "garment_measurements.npz"
RTOL, ATOL = 1e-5, 1e-5

if not MODEL_FILE.exists():
    pytest.skip(f"GarmentMeasurements model not found at {MODEL_PATH}", allow_module_level=True)


def test_load_model_data_has_fbx_skeleton_assets() -> None:
    data = io.load_model_data(MODEL_PATH)

    assert data["components"].shape[:2] == data["mean_vertices"].shape
    assert data["eigenvalues"].shape == (data["components"].shape[-1],)
    assert data["faces"].ndim == 2
    assert data["faces"].shape[1] == 3
    assert len(data["joint_names"]) == data["parents"].shape[0]
    assert data["parents"][0] == -1
    assert data["bind_quats"].shape == (len(data["joint_names"]), 4)
    assert data["skin_weights"].shape == (data["mean_vertices"].shape[0], len(data["joint_names"]))
    assert data["mvc_weights"].shape == data["skin_weights"].shape


def test_fbx_rig_is_in_pca_coordinate_frame() -> None:
    data = io.load_model_data(MODEL_PATH)

    mesh_span = np.ptp(data["mean_vertices"], axis=0)
    joint_positions = data["mvc_weights"].T @ data["mean_vertices"]
    skeleton_span = np.ptp(joint_positions, axis=0)

    assert mesh_span[1] > mesh_span[2]
    assert skeleton_span[1] > skeleton_span[2]


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_get_rest_pose_uses_singleton_shape_batch(backend: str) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(f"body_models.garment_measurements.{backend}")
    model = module.GarmentMeasurements(model_path=MODEL_PATH)
    params = model.get_rest_pose(batch_size=3)

    assert params["shape"].shape == (1, model.num_shape_components)
    assert params["pose"].shape[0] == 3
    assert params["global_rotation"].shape[0] == 3
    assert params["global_translation"].shape == (3, 3)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backends_evaluate_posed_model_consistently(backend: str) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    module = import_module(f"body_models.garment_measurements.{backend}")
    model = module.GarmentMeasurements(model_path=MODEL_PATH)

    shape_np = np.zeros((2, model.num_shape_components), dtype=np.float32)
    shape_np[0, :2] = [0.25, -0.5]
    shape_np[1, 0] = 1.0
    pose_np = np.zeros((2, model.num_joints, 3), dtype=np.float32)
    pose_np[1, 1, 2] = 0.2
    rotation_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    translation_np = np.array([[0.1, 0.2, 0.3], [-0.5, 0.0, 0.5]], dtype=np.float32)

    if backend == "torch":
        import torch

        shape = torch.as_tensor(shape_np)
        pose = torch.as_tensor(pose_np)
        rotation = torch.as_tensor(rotation_np)
        translation = torch.as_tensor(translation_np)
    elif backend == "jax":
        import jax.numpy as jnp

        shape = jnp.asarray(shape_np)
        pose = jnp.asarray(pose_np)
        rotation = jnp.asarray(rotation_np)
        translation = jnp.asarray(translation_np)
    else:
        shape = shape_np
        pose = pose_np
        rotation = rotation_np
        translation = translation_np

    vertices = model.forward_vertices(shape=shape, pose=pose, global_rotation=rotation, global_translation=translation)
    subset = model.forward_vertices(
        shape=shape,
        pose=pose,
        global_rotation=rotation,
        global_translation=translation,
        vertex_indices=[3, 1],
    )
    skeleton = model.forward_skeleton(shape=shape, pose=pose, global_rotation=rotation, global_translation=translation)

    assert np.asarray(vertices).shape == (2, model.num_vertices, 3)
    assert np.asarray(subset).shape == (2, 2, 3)
    assert np.asarray(skeleton).shape == (2, model.num_joints, 4, 4)
    np.testing.assert_allclose(np.asarray(subset), np.asarray(vertices)[:, [3, 1]], atol=ATOL, rtol=RTOL)


def test_numba_backend_matches_numpy() -> None:
    pytest.importorskip("numba")
    from body_models.garment_measurements.numpy import GarmentMeasurements

    numpy_model = GarmentMeasurements(model_path=MODEL_PATH)
    numba_model = GarmentMeasurements(model_path=MODEL_PATH, backend="numba")
    params = numpy_model.get_rest_pose(batch_size=2)
    params["shape"] = np.zeros((2, numpy_model.num_shape_components), dtype=np.float32)
    params["shape"][1, 0] = 0.5
    params["pose"][1, 1, 2] = 0.2
    vertex_indices = [3, 1, 3]

    numpy_vertices = numpy_model.forward_vertices(**params)
    numba_vertices = numba_model.forward_vertices(**params)
    numpy_subset = numpy_model.forward_vertices(**params, vertex_indices=vertex_indices)
    numba_subset = numba_model.forward_vertices(**params, vertex_indices=vertex_indices)

    np.testing.assert_allclose(numba_vertices, numpy_vertices, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(numba_subset, numpy_subset, atol=ATOL, rtol=RTOL)


@pytest.fixture
def model_float64():
    """Create GarmentMeasurements model in float64 for gradient checking."""
    from body_models.garment_measurements.torch import GarmentMeasurements

    return GarmentMeasurements(model_path=MODEL_PATH).to(torch.float64).eval()


def test_gradients_forward_vertices(model_float64) -> None:
    """Test gradients flow correctly through forward_vertices."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_vertices(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


def test_gradients_forward_skeleton(model_float64) -> None:
    """Test gradients flow correctly through forward_skeleton."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_skeleton(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=32)
