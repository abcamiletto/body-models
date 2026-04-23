"""Tests for the SOMA body model."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from body_models.soma.io import SOMA_CORRECTIVES_ASSET, _load_sparse_checkpoint_numpy, get_model_path


@pytest.fixture(scope="module")
def model_path() -> Path:
    return get_model_path()


def _backend_model(backend: str, model_path: Path):
    if backend == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")
    module = import_module(f"body_models.soma.{backend}")
    return getattr(module, "SOMA")(model_path=model_path)


def test_backend_modules_only_export_soma() -> None:
    for backend in ("torch", "numpy", "jax"):
        if backend == "jax":
            pytest.importorskip("jax")
            pytest.importorskip("flax")

        module = import_module(f"body_models.soma.{backend}")
        public_names = sorted(name for name in vars(module) if not name.startswith("_"))
        assert public_names == ["SOMA"]


def test_backends_match(model_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    pytest.importorskip("flax")

    rng = np.random.default_rng(0)
    shape = (rng.standard_normal((1, 128)) * 0.1).astype(np.float32)
    pose = (rng.standard_normal((1, 77, 3)) * 0.05).astype(np.float32)
    global_rotation = (rng.standard_normal((1, 3)) * 0.02).astype(np.float32)
    global_translation = rng.standard_normal((1, 3)).astype(np.float32) * 0.01

    model_torch = _backend_model("torch", model_path)
    model_numpy = _backend_model("numpy", model_path)
    model_jax = _backend_model("jax", model_path)

    verts_torch = np.asarray(
        model_torch.forward_vertices(
            shape=model_torch.mean_active.new_tensor(shape),
            pose=model_torch.mean_active.new_tensor(pose),
            global_rotation=model_torch.mean_active.new_tensor(global_rotation),
            global_translation=model_torch.mean_active.new_tensor(global_translation),
        ).detach()
    )
    skel_torch = np.asarray(
        model_torch.forward_skeleton(
            shape=model_torch.mean_active.new_tensor(shape),
            pose=model_torch.mean_active.new_tensor(pose),
            global_rotation=model_torch.mean_active.new_tensor(global_rotation),
            global_translation=model_torch.mean_active.new_tensor(global_translation),
        ).detach()
    )

    verts_numpy = model_numpy.forward_vertices(
        shape=shape,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )
    skel_numpy = model_numpy.forward_skeleton(
        shape=shape,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
    )

    import jax.numpy as jnp

    verts_jax = np.asarray(
        model_jax.forward_vertices(
            shape=jnp.asarray(shape),
            pose=jnp.asarray(pose),
            global_rotation=jnp.asarray(global_rotation),
            global_translation=jnp.asarray(global_translation),
        )
    )
    skel_jax = np.asarray(
        model_jax.forward_skeleton(
            shape=jnp.asarray(shape),
            pose=jnp.asarray(pose),
            global_rotation=jnp.asarray(global_rotation),
            global_translation=jnp.asarray(global_translation),
        )
    )

    np.testing.assert_allclose(verts_torch, verts_numpy, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(verts_torch, verts_jax, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(skel_torch, skel_numpy, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(skel_torch, skel_jax, atol=1e-5, rtol=1e-5)


def test_simplify_reduces_mesh(model_path: Path) -> None:
    from body_models.soma.torch import SOMA

    model_full = SOMA(model_path=model_path, simplify=1.0)
    model_half = SOMA(model_path=model_path, simplify=2.0)

    assert model_half.num_vertices < model_full.num_vertices
    assert model_half.faces.shape[0] < model_full.faces.shape[0]

    params = model_half.get_rest_pose(batch_size=2)
    verts = model_half.forward_vertices(**params)
    skel = model_half.forward_skeleton(**params)

    assert verts.shape == (2, model_half.num_vertices, 3)
    assert skel.shape == (2, model_half.num_joints, 4, 4)


def test_apply_correctives_requires_weights(model_path: Path) -> None:
    from body_models.soma.numpy import SOMA

    model = SOMA(model_path=model_path)
    params = model.get_rest_pose()
    model.corrective_W1 = None

    with pytest.raises(ValueError, match="apply_correctives=True requires SOMA corrective weights."):
        model.forward_vertices(**params)

    verts = model.forward_vertices(**params, apply_correctives=False)
    assert verts.shape == (1, model.num_vertices, 3)


def test_corrective_checkpoint_loads_with_ptloader(model_path: Path) -> None:
    checkpoint = _load_sparse_checkpoint_numpy(model_path / SOMA_CORRECTIVES_ASSET)

    assert "bindpose" in checkpoint
    assert "W1" in checkpoint
    assert "W2" in checkpoint
