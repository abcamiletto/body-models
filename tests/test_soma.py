"""Tests for the SOMA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from body_models.soma.io import get_model_path

MODEL_TYPE_SEEDS = {
    "soma": 11,
    "mhr": 22,
    "smpl": 33,
    "smplx": 44,
}


@pytest.fixture(scope="module")
def model_path() -> Path:
    return get_model_path()


def _backend_model(backend: str, model_path: Path, **kwargs):
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")
    module = import_module(f"body_models.soma.{backend}")
    return getattr(module, "SOMA")(model_path=model_path, **kwargs)


def _sample_inputs(model_type: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(MODEL_TYPE_SEEDS[model_type])
    inputs = {
        "pose": (rng.standard_normal((1, 77, 3)) * 0.05).astype(np.float32),
        "global_rotation": (rng.standard_normal((1, 3)) * 0.02).astype(np.float32),
        "global_translation": (rng.standard_normal((1, 3)) * 0.01).astype(np.float32),
    }
    if model_type == "soma":
        inputs["identity"] = (rng.standard_normal((1, 128)) * 0.1).astype(np.float32)
    elif model_type == "mhr":
        inputs["identity"] = (rng.standard_normal((1, 45)) * 0.1).astype(np.float32)
        inputs["scale_params"] = (rng.standard_normal((1, 68)) * 0.05).astype(np.float32)
    else:
        inputs["identity"] = (rng.standard_normal((1, 10)) * 0.1).astype(np.float32)
    return inputs


@pytest.mark.parametrize("model_type", ["soma", "mhr", "smpl", "smplx"])
def test_backends_match(model_type: str, model_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    inputs = _sample_inputs(model_type)
    model_torch = _backend_model("torch", model_path, model_type=model_type)
    model_numpy = _backend_model("numpy", model_path, model_type=model_type)
    model_jax = _backend_model("jax", model_path, model_type=model_type)

    torch_inputs = {name: model_torch.mean_active.new_tensor(value) for name, value in inputs.items()}
    verts_torch = np.asarray(model_torch.forward_vertices(**torch_inputs).detach())
    skel_torch = np.asarray(model_torch.forward_skeleton(**torch_inputs).detach())

    verts_numpy = model_numpy.forward_vertices(**inputs)
    skel_numpy = model_numpy.forward_skeleton(**inputs)

    import jax.numpy as jnp

    jax_inputs = {name: jnp.asarray(value) for name, value in inputs.items()}
    verts_jax = np.asarray(model_jax.forward_vertices(**jax_inputs))
    skel_jax = np.asarray(model_jax.forward_skeleton(**jax_inputs))

    np.testing.assert_allclose(verts_torch, verts_numpy, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(verts_torch, verts_jax, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(skel_torch, skel_numpy, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(skel_torch, skel_jax, atol=1e-4, rtol=1e-4)


def test_simplify_reduces_mesh(model_path: Path) -> None:
    from body_models.soma.torch import SOMA

    model_full = SOMA(model_path=model_path, simplify=1.0)
    model_half = SOMA(model_path=model_path, simplify=2.0)

    assert model_half.num_vertices < model_full.num_vertices
    assert model_half.faces.shape[0] < model_full.faces.shape[0]

    params = model_half.get_rest_pose(batch_size=2)
    verts = model_half.forward_vertices(
        identity=params["identity"],
        pose=params["pose"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
    )
    skel = model_half.forward_skeleton(
        identity=params["identity"],
        pose=params["pose"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
    )

    assert verts.shape == (2, model_half.num_vertices, 3)
    assert skel.shape == (2, model_half.num_joints, 4, 4)
