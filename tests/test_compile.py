"""Tests for model compilation in torch and JAX.

Verifies that all models can be compiled with torch.compile and jax.jit,
producing the same results as uncompiled execution and without graph breaks.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

ASSET_DIR = Path(__file__).parent / "assets"
MODEL_CASES = [
    pytest.param("smpl", {}, id="smpl"),
    pytest.param("smplx", {}, id="smplx"),
    pytest.param("skel", {}, id="skel"),
    pytest.param("flame", {}, id="flame"),
    pytest.param("anny", {}, id="anny"),
    pytest.param("mhr", {}, id="mhr"),
    pytest.param("soma", {"model_type": "soma"}, id="soma"),
    pytest.param("soma", {"model_type": "anny"}, id="soma-anny"),
    pytest.param("soma", {"model_type": "mhr"}, id="soma-mhr"),
]


def get_compile_tolerances(model_name: str) -> tuple[float, float]:
    if model_name == "soma":
        return 1e-4, 1e-4
    return 1e-5, 1e-5


def get_model_file(model_name: str) -> Path:
    """Get the actual model file path for a given model."""
    if model_name == "soma":
        from body_models.soma.io import get_model_path

        return get_model_path()

    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir  # Will trigger skip

    if model_name == "smpl":
        return model_dir / "SMPL_NEUTRAL.npz"
    if model_name == "smplx":
        return model_dir / "SMPLX_NEUTRAL.npz"
    if model_name == "flame":
        return model_dir / "FLAME_NEUTRAL.pkl"
    if model_name == "skel":
        return model_dir / "skel_male.pkl"

    return model_dir


def get_torch_model(model_name: str, model_path: Path, **kwargs):
    """Import and instantiate a PyTorch model by name."""
    if model_name == "smpl":
        from body_models.smpl.torch import SMPL

        return SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.torch import SMPLX

        return SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.torch import SKEL

        return SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.torch import FLAME

        return FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.torch import ANNY

        return ANNY()
    elif model_name == "mhr":
        from body_models.mhr.torch import MHR

        return MHR()
    elif model_name == "soma":
        from body_models.soma.torch import SOMA

        return SOMA(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# PyTorch compilation correctness tests
# ============================================================================


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_torch_compile_forward_vertices(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test torch.compile produces correct results for forward_vertices."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_torch_model(model_name, model_path, **model_kwargs)
    model.eval()

    # Compile model
    compiled_fn = torch.compile(model.forward_vertices)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        # Run uncompiled
        result_eager = model.forward_vertices(**params)

        # Run compiled
        result_compiled = compiled_fn(**params)

    # Verify same results
    rtol, atol = get_compile_tolerances(model_name)
    np.testing.assert_allclose(result_compiled.numpy(), result_eager.numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_torch_compile_forward_skeleton(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test torch.compile produces correct results for forward_skeleton."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_torch_model(model_name, model_path, **model_kwargs)
    model.eval()

    # Compile model
    compiled_fn = torch.compile(model.forward_skeleton)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        # Run uncompiled
        result_eager = model.forward_skeleton(**params)

        # Run compiled
        result_compiled = compiled_fn(**params)

    # Verify same results
    rtol, atol = get_compile_tolerances(model_name)
    np.testing.assert_allclose(result_compiled.numpy(), result_eager.numpy(), rtol=rtol, atol=atol)


# ============================================================================
# PyTorch fullgraph compilation tests (no graph breaks)
# ============================================================================


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_torch_compile_fullgraph_forward_vertices(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test torch.compile with fullgraph=True (no graph breaks) for forward_vertices."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_torch_model(model_name, model_path, **model_kwargs)
    model.eval()

    # Compile with fullgraph=True - will fail if there are any graph breaks
    compiled_fn = torch.compile(model.forward_vertices, fullgraph=True)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        # Run compiled - this will raise if graph breaks occur during tracing
        result_compiled = compiled_fn(**params)

    # Basic sanity check
    assert result_compiled.shape[0] == 2
    assert result_compiled.shape[2] == 3


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_torch_compile_fullgraph_forward_skeleton(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test torch.compile with fullgraph=True (no graph breaks) for forward_skeleton."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_torch_model(model_name, model_path, **model_kwargs)
    model.eval()

    # Compile with fullgraph=True - will fail if there are any graph breaks
    compiled_fn = torch.compile(model.forward_skeleton, fullgraph=True)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        # Run compiled - this will raise if graph breaks occur during tracing
        result_compiled = compiled_fn(**params)

    # Basic sanity check
    assert result_compiled.shape[0] == 2
    assert result_compiled.shape[2:] == (4, 4)


# ============================================================================
# JAX compilation tests
# ============================================================================


def get_jax_model(model_name: str, model_path: Path, **kwargs):
    """Import and instantiate a JAX model by name."""
    if model_name == "smpl":
        from body_models.smpl.jax import SMPL

        return SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.jax import SMPLX

        return SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.jax import SKEL

        return SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.jax import FLAME

        return FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.jax import ANNY

        return ANNY()
    elif model_name == "mhr":
        from body_models.mhr.jax import MHR

        return MHR()
    elif model_name == "soma":
        from body_models.soma.jax import SOMA

        return SOMA(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_jax_jit_forward_vertices(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test jax.jit produces correct results for forward_vertices."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")

    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_jax_model(model_name, model_path, **model_kwargs)

    # JIT compile
    jitted_fn = jax.jit(model.forward_vertices)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    # Run unjitted
    result_eager = model.forward_vertices(**params)

    # Run jitted (first call compiles, second uses cache)
    result_jitted = jitted_fn(**params)
    result_jitted_2 = jitted_fn(**params)

    # Verify same results
    rtol, atol = get_compile_tolerances(model_name)
    np.testing.assert_allclose(np.asarray(result_jitted), np.asarray(result_eager), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(result_jitted_2), np.asarray(result_eager), rtol=rtol, atol=atol)


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_jax_jit_forward_skeleton(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test jax.jit produces correct results for forward_skeleton."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")

    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    model = get_jax_model(model_name, model_path, **model_kwargs)

    # JIT compile
    jitted_fn = jax.jit(model.forward_skeleton)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    # Run unjitted
    result_eager = model.forward_skeleton(**params)

    # Run jitted
    result_jitted = jitted_fn(**params)
    result_jitted_2 = jitted_fn(**params)

    # Verify same results
    rtol, atol = get_compile_tolerances(model_name)
    np.testing.assert_allclose(np.asarray(result_jitted), np.asarray(result_eager), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(result_jitted_2), np.asarray(result_eager), rtol=rtol, atol=atol)
