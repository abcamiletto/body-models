"""Tests for model compilation in torch and JAX.

Verifies that all models can be compiled with torch.compile and jax.jit,
producing the same results as uncompiled execution.

Note: Due to array_api_compat usage for backend-agnostic code, torch.compile
may have graph breaks. These tests verify correctness, not full fusion.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

ASSET_DIR = Path(__file__).parent / "assets"


def get_model_file(model_name: str) -> Path:
    """Get the actual model file path for a given model."""
    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir  # Will trigger skip

    # Find the model file
    for ext in [".npz", ".pkl"]:
        for f in model_dir.glob(f"*{ext}"):
            return f

    return model_dir


# ============================================================================
# PyTorch compilation tests
# ============================================================================


@pytest.mark.parametrize("model_name", ["smpl", "smplx", "skel", "flame", "anny", "mhr"])
def test_torch_compile_forward_vertices(model_name: str) -> None:
    """Test torch.compile produces correct results for forward_vertices."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    # Import and instantiate model
    if model_name == "smpl":
        from body_models.smpl.torch import SMPL

        model = SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.torch import SMPLX

        model = SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.torch import SKEL

        model = SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.torch import FLAME

        model = FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.torch import ANNY

        model = ANNY()
    elif model_name == "mhr":
        from body_models.mhr.torch import MHR

        model = MHR()

    model.eval()

    # Compile model (allow graph breaks due to array_api_compat)
    compiled_fn = torch.compile(model.forward_vertices)

    # Get test params
    params = model.get_rest_pose(batch_size=2)

    with torch.no_grad():
        # Run uncompiled
        result_eager = model.forward_vertices(**params)

        # Run compiled
        result_compiled = compiled_fn(**params)

    # Verify same results
    np.testing.assert_allclose(result_compiled.numpy(), result_eager.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_name", ["smpl", "smplx", "skel", "flame", "anny", "mhr"])
def test_torch_compile_forward_skeleton(model_name: str) -> None:
    """Test torch.compile produces correct results for forward_skeleton."""
    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    # Import and instantiate model
    if model_name == "smpl":
        from body_models.smpl.torch import SMPL

        model = SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.torch import SMPLX

        model = SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.torch import SKEL

        model = SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.torch import FLAME

        model = FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.torch import ANNY

        model = ANNY()
    elif model_name == "mhr":
        from body_models.mhr.torch import MHR

        model = MHR()

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
    np.testing.assert_allclose(result_compiled.numpy(), result_eager.numpy(), rtol=1e-5, atol=1e-5)


# ============================================================================
# JAX compilation tests
# ============================================================================


@pytest.mark.parametrize("model_name", ["smpl", "smplx", "skel", "flame", "anny", "mhr"])
def test_jax_jit_forward_vertices(model_name: str) -> None:
    """Test jax.jit produces correct results for forward_vertices."""
    jax = pytest.importorskip("jax")

    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    # Import and instantiate model
    if model_name == "smpl":
        from body_models.smpl.jax import SMPL

        model = SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.jax import SMPLX

        model = SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.jax import SKEL

        model = SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.jax import FLAME

        model = FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.jax import ANNY

        model = ANNY()
    elif model_name == "mhr":
        from body_models.mhr.jax import MHR

        model = MHR()

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
    np.testing.assert_allclose(np.asarray(result_jitted), np.asarray(result_eager), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result_jitted_2), np.asarray(result_eager), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_name", ["smpl", "smplx", "skel", "flame", "anny", "mhr"])
def test_jax_jit_forward_skeleton(model_name: str) -> None:
    """Test jax.jit produces correct results for forward_skeleton."""
    jax = pytest.importorskip("jax")

    model_path = get_model_file(model_name)
    if not model_path.exists():
        pytest.skip(f"Model assets not found: {model_path}")

    # Import and instantiate model
    if model_name == "smpl":
        from body_models.smpl.jax import SMPL

        model = SMPL(model_path=model_path)
    elif model_name == "smplx":
        from body_models.smplx.jax import SMPLX

        model = SMPLX(model_path=model_path)
    elif model_name == "skel":
        from body_models.skel.jax import SKEL

        model = SKEL(gender="male", model_path=model_path)
    elif model_name == "flame":
        from body_models.flame.jax import FLAME

        model = FLAME(model_path=model_path)
    elif model_name == "anny":
        from body_models.anny.jax import ANNY

        model = ANNY()
    elif model_name == "mhr":
        from body_models.mhr.jax import MHR

        model = MHR()

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
    np.testing.assert_allclose(np.asarray(result_jitted), np.asarray(result_eager), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(result_jitted_2), np.asarray(result_eager), rtol=1e-5, atol=1e-5)
