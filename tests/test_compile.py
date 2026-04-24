"""Tests for model compilation in torch and JAX.

Verifies that all models can be compiled with torch.compile and jax.jit,
producing the same results as uncompiled execution and without graph breaks.
"""

import gc
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

ASSET_DIR = Path(__file__).parent / "assets"
MODEL_FILES = {
    "smpl": "SMPL_NEUTRAL.npz",
    "smplx": "SMPLX_NEUTRAL.npz",
    "flame": "FLAME_NEUTRAL.pkl",
    "skel": "skel_male.pkl",
}
CLASS_NAMES = {name: ("FLAME" if name == "flame" else name.upper()) for name in (*MODEL_FILES, "anny", "mhr", "soma")}
CLASS_NAMES["garment_measurements"] = "GarmentMeasurements"
MODEL_CASES = (
    pytest.param("smpl", {}, id="smpl"),
    pytest.param("smplx", {}, id="smplx"),
    pytest.param("skel", {}, id="skel"),
    pytest.param("flame", {}, id="flame"),
    pytest.param("anny", {}, id="anny"),
    pytest.param("mhr", {}, id="mhr"),
    pytest.param("soma", {"model_type": "soma"}, id="soma"),
    pytest.param("soma", {"model_type": "anny"}, id="soma-anny"),
    pytest.param("soma", {"model_type": "mhr"}, id="soma-mhr"),
    pytest.param("soma", {"model_type": "smpl"}, id="soma-smpl"),
    pytest.param("soma", {"model_type": "smplx"}, id="soma-smplx"),
    pytest.param("garment_measurements", {}, id="garment-measurements"),
)
COMPILE_TOLERANCES = {"soma": (1e-4, 1e-4)}


def get_compile_tolerances(model_name: str) -> tuple[float, float]:
    return COMPILE_TOLERANCES.get(model_name, (1e-5, 1e-5))


def get_model_file(model_name: str) -> Path:
    """Get the actual model file path for a given model."""
    if model_name == "garment_measurements":
        return ASSET_DIR / "garment_measurements" / "model" / "garment_measurements.npz"

    if model_name == "soma":
        from body_models.soma.io import get_model_path

        return get_model_path()

    model_dir = ASSET_DIR / model_name / "model"
    if not model_dir.exists():
        return model_dir  # Will trigger skip

    filename = MODEL_FILES.get(model_name)
    return model_dir if filename is None else model_dir / filename


def get_required_model_files(model_name: str, model_kwargs: dict[str, str]) -> list[Path]:
    paths = [get_model_file(model_name)]
    if model_name != "soma":
        return paths

    nested_model_type = model_kwargs.get("model_type")
    if nested_model_type in MODEL_FILES:
        paths.append(ASSET_DIR / nested_model_type / "model" / MODEL_FILES[nested_model_type])
    return paths


def get_model(backend: str, model_name: str, model_path: Path, **kwargs: Any) -> Any:
    module = import_module(f"body_models.{model_name}.{backend}")
    model_class = getattr(module, CLASS_NAMES[model_name])
    ctor_kwargs = {"model_path": model_path, **kwargs}
    if model_name == "skel":
        ctor_kwargs["gender"] = "male"
    return model_class(**ctor_kwargs)


@pytest.fixture(autouse=True)
def clear_compile_caches() -> None:
    yield
    gc.collect()
    torch._dynamo.reset()
    try:
        import jax
    except ImportError:
        return
    jax.clear_caches()


# ============================================================================
# PyTorch compilation correctness tests
# ============================================================================


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_torch_compile_forward_vertices(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test torch.compile produces correct results for forward_vertices."""
    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("torch", model_name, model_path, **model_kwargs)
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
    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("torch", model_name, model_path, **model_kwargs)
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
    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("torch", model_name, model_path, **model_kwargs)
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
    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("torch", model_name, model_path, **model_kwargs)
    if not hasattr(model, "forward_skeleton"):
        pytest.skip(f"{model_name} does not expose forward_skeleton")
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


@pytest.mark.parametrize(("model_name", "model_kwargs"), MODEL_CASES)
def test_jax_jit_forward_vertices(model_name: str, model_kwargs: dict[str, str]) -> None:
    """Test jax.jit produces correct results for forward_vertices."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")

    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("jax", model_name, model_path, **model_kwargs)

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

    required_paths = get_required_model_files(model_name, model_kwargs)
    missing_path = next((path for path in required_paths if not path.exists()), None)
    if missing_path is not None:
        pytest.skip(f"Model assets not found: {missing_path}")

    model_path = required_paths[0]
    model = get_model("jax", model_name, model_path, **model_kwargs)
    if not hasattr(model, "forward_skeleton"):
        pytest.skip(f"{model_name} does not expose forward_skeleton")

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
