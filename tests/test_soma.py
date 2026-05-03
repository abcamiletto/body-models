"""Tests for the SOMA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from body_models.soma.io import get_model_path

pytestmark = pytest.mark.fast

MODEL_TYPE_SEEDS = {
    "soma": 11,
    "anny": 17,
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
    elif model_type == "anny":
        inputs["identity"] = rng.uniform(0.2, 0.8, size=(1, 6)).astype(np.float32)
    elif model_type == "mhr":
        inputs["identity"] = (rng.standard_normal((1, 45)) * 0.1).astype(np.float32)
        inputs["scale_params"] = (rng.standard_normal((1, 68)) * 0.05).astype(np.float32)
    else:
        inputs["identity"] = (rng.standard_normal((1, 10)) * 0.1).astype(np.float32)
    return inputs


@pytest.mark.parametrize("model_type", ["soma", "anny", "mhr", "smpl", "smplx"])
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


def test_numpy_prepare_identity_matches_forward(model_path: Path) -> None:
    from body_models.soma.numpy import SOMA

    model = SOMA(model_path=model_path)
    params = model.get_rest_pose(batch_size=4)
    prepared_identity = model.prepare_identity(identity=params["identity"])

    vertices = model.forward_vertices(
        pose=params["pose"],
        identity=params["identity"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
    )
    skeleton = model.forward_skeleton(
        pose=params["pose"],
        identity=params["identity"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
    )
    prepared_vertices = model.forward_vertices(
        pose=params["pose"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
        prepared_identity=prepared_identity,
    )
    prepared_skeleton = model.forward_skeleton(
        pose=params["pose"],
        global_rotation=params["global_rotation"],
        global_translation=params["global_translation"],
        prepared_identity=prepared_identity,
    )

    np.testing.assert_allclose(prepared_vertices, vertices, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(prepared_skeleton, skeleton, atol=1e-5, rtol=1e-5)


def test_mhr_rotmat_backward_without_correctives(model_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from body_models.soma.torch import SOMA

    model = SOMA(model_path=model_path, model_type="mhr", rotation_type="rotmat")

    batch_size = 4
    pose = torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, model.num_joints, 1, 1)
    global_translation = torch.zeros(batch_size, 3)

    torch.manual_seed(0)
    identity0 = torch.randn(1, model.identity_dim) * 0.01
    scale0 = torch.randn(1, model.num_scale_params) * 0.01

    with torch.no_grad():
        target = model.forward_vertices(
            pose=pose,
            identity=identity0,
            scale_params=scale0,
            global_translation=global_translation,
            apply_correctives=False,
        ).detach()

    identity = torch.nn.Parameter(identity0.clone())
    scale_params = torch.nn.Parameter(scale0.clone())
    pred = model.forward_vertices(
        pose=pose.clone(),
        identity=identity,
        scale_params=scale_params,
        global_translation=global_translation,
        apply_correctives=False,
    )

    loss = (pred - target).square().mean()
    loss.backward()

    assert identity.grad is not None
    assert scale_params.grad is not None
