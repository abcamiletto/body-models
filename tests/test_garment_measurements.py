"""Tests for the GarmentMeasurements PCA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

from body_models.garment_measurements import io
from gradient_utils import prepare_params, sampled_gradcheck

ASSET_DIR = Path(__file__).parent / "assets" / "garment_measurements"
MODEL_PATH = ASSET_DIR / "model"
RTOL, ATOL = 1e-5, 1e-5

if not (MODEL_PATH / "pca" / "point.pca").exists():
    pytest.skip(f"GarmentMeasurements model not found at {MODEL_PATH}", allow_module_level=True)


def test_load_model_data_matches_upstream_files() -> None:
    data = io.load_model_data(MODEL_PATH)
    obj_vertices, faces = io.load_obj_mesh(MODEL_PATH / "pca" / "mean.obj")

    assert data["components"].shape[:2] == data["mean_vertices"].shape
    assert data["eigenvalues"].shape == (data["components"].shape[-1],)
    np.testing.assert_allclose(data["mean_vertices"], obj_vertices, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(data["faces"], faces)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backends_match_original_pca_evaluation(backend: str) -> None:
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
    rotation_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    translation_np = np.array([[0.1, 0.2, 0.3], [-0.5, 0.0, 0.5]], dtype=np.float32)
    expected = np.asarray(model.mean_vertices)[None] + np.einsum(
        "bc,vdc->bvd",
        shape_np * np.sqrt(np.asarray(model.eigenvalues))[None],
        np.asarray(model.components),
    )
    expected[1] = expected[1] @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    expected = expected + translation_np[:, None]

    if backend == "torch":
        import torch

        shape = torch.as_tensor(shape_np)
        rotation = torch.as_tensor(rotation_np)
        translation = torch.as_tensor(translation_np)
    elif backend == "jax":
        import jax.numpy as jnp

        shape = jnp.asarray(shape_np)
        rotation = jnp.asarray(rotation_np)
        translation = jnp.asarray(translation_np)
    else:
        shape = shape_np
        rotation = rotation_np
        translation = translation_np

    vertices = model.forward_vertices(shape=shape, global_rotation=rotation, global_translation=translation)
    subset = model.forward_vertices(
        shape=shape,
        global_rotation=rotation,
        global_translation=translation,
        vertex_indices=[3, 1],
    )

    assert not hasattr(model, "forward_skeleton")
    np.testing.assert_allclose(np.asarray(vertices), expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(np.asarray(subset), expected[:, [3, 1]], atol=ATOL, rtol=RTOL)


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


def test_forward_skeleton_is_not_exposed(model_float64) -> None:
    assert not hasattr(model_float64, "forward_skeleton")
