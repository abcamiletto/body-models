"""Tests for the SKEL body model.

- Numerical precision: parametrized across torch/numpy/jax backends
- Gradient correctness: torch backend with gradcheck
- Feature tests: mesh simplification, etc.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from accelerator_utils import get_accelerator_device
from gradient_utils import prepare_params, sampled_gradcheck

ASSET_DIR = Path(__file__).parent / "assets" / "skel"
MODEL_PATH = ASSET_DIR / "model"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5
RTOL, ATOL = 1e-4, 1e-4

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"SKEL model not found at {MODEL_PATH}",
)


# ============================================================================
# Test data loading
# ============================================================================


def load_test_case(idx: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load inputs and reference outputs for a test case."""
    data = json.loads((INPUTS_DIR / f"{idx}.json").read_text())
    inputs = {
        "gender": data["gender"],
        "shape": np.array(data["shape"], dtype=np.float32),
        "body_pose": np.array(data["body_pose"], dtype=np.float32),
        "global_translation": np.array(data["trans"], dtype=np.float32),
    }
    outputs = {
        "vertices": np.load(OUTPUTS_DIR / str(idx) / "vertices.npy"),
        "joints": np.load(OUTPUTS_DIR / str(idx) / "joints.npy"),
    }
    return inputs, outputs


# ============================================================================
# Numerical precision tests (all backends)
# ============================================================================


@requires_model
@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_torch(idx: int) -> None:
    """Test PyTorch forward_vertices matches reference."""
    from body_models.skel.torch import SKEL, from_native_args, to_native_outputs

    inputs, ref = load_test_case(idx)
    model = SKEL(gender=inputs["gender"], model_path=MODEL_PATH)
    model.eval()

    args = from_native_args(
        shape=torch.tensor(inputs["shape"])[None],
        body_pose=torch.tensor(inputs["body_pose"])[None],
        global_translation=torch.tensor(inputs["global_translation"])[None],
    )

    with torch.no_grad():
        verts = model.forward_vertices(**args)
        transforms = model.forward_skeleton(**args)
        skel_mesh = model.forward_skeleton_mesh(**args)

    result = to_native_outputs(verts, transforms, skel_mesh, model._feet_offset)
    np.testing.assert_allclose(result["vertices"][0].numpy(), ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result["joints"][0].numpy(), ref["joints"], rtol=RTOL, atol=ATOL)


@requires_model
@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_numpy(idx: int) -> None:
    """Test NumPy forward_vertices matches reference."""
    from body_models.skel.numpy import SKEL, from_native_args, to_native_outputs

    inputs, ref = load_test_case(idx)
    model = SKEL(gender=inputs["gender"], model_path=MODEL_PATH)

    args = from_native_args(
        shape=inputs["shape"][None],
        body_pose=inputs["body_pose"][None],
        global_translation=inputs["global_translation"][None],
    )

    verts = model.forward_vertices(**args)
    transforms = model.forward_skeleton(**args)
    result = to_native_outputs(verts, transforms, model._feet_offset)

    np.testing.assert_allclose(result["vertices"][0], ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result["joints"][0], ref["joints"], rtol=RTOL, atol=ATOL)


@requires_model
@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_jax(idx: int) -> None:
    """Test JAX forward_vertices matches reference."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.skel.jax import SKEL, from_native_args, to_native_outputs

    inputs, ref = load_test_case(idx)
    model = SKEL(gender=inputs["gender"], model_path=MODEL_PATH)

    args = from_native_args(
        shape=jnp.array(inputs["shape"])[None],
        body_pose=jnp.array(inputs["body_pose"])[None],
        global_translation=jnp.array(inputs["global_translation"])[None],
    )

    verts = model.forward_vertices(**args)
    transforms = model.forward_skeleton(**args)
    result = to_native_outputs(verts, transforms, model._feet_offset[...])

    np.testing.assert_allclose(np.asarray(result["vertices"][0]), ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(result["joints"][0]), ref["joints"], rtol=RTOL, atol=ATOL)


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create SKEL model in float64 for gradient checking."""
    from body_models.skel.torch import SKEL

    if not MODEL_PATH.exists():
        pytest.skip(f"SKEL model not found at {MODEL_PATH}")

    model = SKEL(model_path=MODEL_PATH, gender="male")
    return model.to(torch.float64).eval()


@requires_model
def test_gradients_forward_vertices(model_float64) -> None:
    """Test gradients flow correctly through forward_vertices."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_vertices(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


@requires_model
def test_gradients_forward_skeleton(model_float64) -> None:
    """Test gradients flow correctly through forward_skeleton."""
    params = prepare_params(model_float64.get_rest_pose(batch_size=1))
    inputs = tuple(params.values())

    def fn(*tensors):
        kwargs = dict(zip(params.keys(), tensors))
        return model_float64.forward_skeleton(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


# ============================================================================
# Feature tests
# ============================================================================


@requires_model
def test_forward_accelerator_optional_defaults() -> None:
    """Test accelerator forward_* with omitted optional params stays on-device."""
    from body_models.skel.torch import SKEL

    device = get_accelerator_device()
    if device is None:
        pytest.skip("No accelerator available (cuda or mps)")

    model = SKEL(model_path=MODEL_PATH, gender="male").to(device)
    B = 2
    params = model.get_rest_pose(batch_size=B)
    params["pose"] = torch.randn_like(params["pose"]) * 0.1
    params.pop("global_rotation")
    params.pop("global_translation")

    with torch.no_grad():
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)

    assert verts.device.type == device.type
    assert skel.device.type == device.type


@requires_model
def test_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    from body_models.skel.torch import SKEL

    model_orig = SKEL(model_path=MODEL_PATH, gender="male", simplify=1.0)
    model_2x = SKEL(model_path=MODEL_PATH, gender="male", simplify=2.0)
    model_4x = SKEL(model_path=MODEL_PATH, gender="male", simplify=4.0)

    # Check vertex/face counts are reduced
    assert model_2x.num_vertices < model_orig.num_vertices
    assert model_4x.num_vertices < model_2x.num_vertices
    assert model_2x.faces.shape[0] < model_orig.faces.shape[0]
    assert model_4x.faces.shape[0] < model_2x.faces.shape[0]

    # Check approximate ratios (within 10% tolerance)
    assert abs(model_2x.faces.shape[0] / model_orig.faces.shape[0] - 0.5) < 0.1
    assert abs(model_4x.faces.shape[0] / model_orig.faces.shape[0] - 0.25) < 0.1

    # Test forward pass works
    params = model_2x.get_rest_pose(batch_size=2)
    verts = model_2x.forward_vertices(**params)
    skel = model_2x.forward_skeleton(**params)

    assert verts.shape == (2, model_2x.num_vertices, 3)
    assert skel.shape == (2, 24, 4, 4)

    # Skeleton should be nearly identical (uses full-resolution mesh internally)
    params_orig = model_orig.get_rest_pose(batch_size=1)
    joints_orig = model_orig.forward_skeleton(**params_orig)[0, :, :3, 3]
    joints_2x = model_2x.forward_skeleton(**params_orig)[0, :, :3, 3]
    # Allow small numerical tolerance (< 1mm)
    assert (joints_orig - joints_2x).abs().max() < 0.001
