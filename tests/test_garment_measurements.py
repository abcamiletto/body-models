"""Tests for the GarmentMeasurements PCA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import torch

import garment_measurements_asset as asset
from body_models.garment_measurements import io
from gradient_utils import prepare_params, sampled_gradcheck


@pytest.fixture()
def model_fixture() -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (asset.get_garment_measurements_model_path(), *asset.synthetic_garment_measurements_data())


def test_load_pca_matches_upstream_binary_layout(
    model_fixture: tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    model_path, mean, components, eigenvalues, _ = model_fixture
    loaded_mean, loaded_components, loaded_eigenvalues = io.load_pca(model_path / "pca" / "point.pca")

    np.testing.assert_allclose(loaded_mean, mean)
    np.testing.assert_allclose(loaded_components, components)
    np.testing.assert_allclose(loaded_eigenvalues, eigenvalues)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_backends_match_original_pca_evaluation(
    backend: str,
    model_fixture: tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")

    model_path, mean, components, eigenvalues, faces = model_fixture
    module = import_module(f"body_models.garment_measurements.{backend}")
    model = module.GarmentMeasurements(model_path=model_path)

    shape_np = np.array([[0.25, -0.5], [1.0, 0.0]], dtype=np.float32)
    rotation_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    translation_np = np.array([[0.1, 0.2, 0.3], [-0.5, 0.0, 0.5]], dtype=np.float32)
    expected = mean[None] + np.einsum("bc,vdc->bvd", shape_np * np.sqrt(eigenvalues)[None], components)
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
    skeleton = model.forward_skeleton(shape=shape, global_rotation=rotation, global_translation=translation)

    assert model.num_joints == 1
    assert model.joint_names == ["root"]
    assert model.parents == [-1]
    assert model.skin_weights.shape == (4, 1)
    np.testing.assert_array_equal(np.asarray(model.faces), faces)
    np.testing.assert_allclose(np.asarray(vertices), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(subset), expected[:, [3, 1]], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(skeleton)[1, 0, 0, 1], -1.0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(skeleton)[:, 0, :3, 3], translation_np, atol=1e-6, rtol=1e-6)


@pytest.fixture
def model_float64(model_fixture: tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Create GarmentMeasurements model in float64 for gradient checking."""
    from body_models.garment_measurements.torch import GarmentMeasurements

    model_path, *_ = model_fixture
    return GarmentMeasurements(model_path=model_path).to(torch.float64).eval()


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

    assert sampled_gradcheck(fn, inputs, n_samples=64)
