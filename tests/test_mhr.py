"""Tests for the MHR (Meta Human Rig) body model.

- Numerical precision: parametrized across torch/numpy/jax backends
- Gradient correctness: torch backend with gradcheck
- Feature tests: mesh simplification, etc.

Note: Reference outputs are in centimeters (MHR native units), while the model
outputs meters. Tests convert model output to cm for comparison.

All backends (PyTorch, NumPy, JAX) include neural pose correctives.
Tests compare against reference outputs with pose correctives.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from accelerator_utils import get_accelerator_device
from gradient_utils import prepare_params, sampled_gradcheck

ASSET_DIR = Path(__file__).parent / "assets" / "mhr"
MODEL_PATH = ASSET_DIR / "model"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5
RTOL, ATOL = 1e-4, 1e-4


# ============================================================================
# Test data loading
# ============================================================================


def load_test_case(idx: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load inputs and reference outputs for a test case."""
    data = json.loads((INPUTS_DIR / f"{idx}.json").read_text())
    inputs = {
        "shape": np.array(data["shape"], dtype=np.float32),
        "expression": np.array(data["expression"], dtype=np.float32),
        "pose": np.array(data["pose"], dtype=np.float32),
    }
    outputs = {
        "vertices": np.load(OUTPUTS_DIR / str(idx) / "vertices.npy"),
        "skeleton": np.load(OUTPUTS_DIR / str(idx) / "skeleton.npy"),
    }
    return inputs, outputs


def _assert_skeleton_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Compare skeleton states, handling quaternion double cover (q ~ -q)."""
    # Format: [J, 8] = [translation(3), quaternion(4), scale(1)]
    t_actual, q_actual, s_actual = actual[:, :3], actual[:, 3:7], actual[:, 7:]
    t_expected, q_expected, s_expected = expected[:, :3], expected[:, 3:7], expected[:, 7:]

    torch.testing.assert_close(t_actual, t_expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(s_actual, s_expected, rtol=rtol, atol=atol)

    # Quaternions: q and -q represent the same rotation
    q_diff_pos = (q_actual - q_expected).abs().max(dim=-1).values
    q_diff_neg = (q_actual + q_expected).abs().max(dim=-1).values
    q_diff = torch.minimum(q_diff_pos, q_diff_neg)
    assert (q_diff < atol).all(), f"Quaternion mismatch: max diff {q_diff.max().item()}"


# ============================================================================
# Numerical precision tests (all backends)
# ============================================================================


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_torch(idx: int) -> None:
    """Test PyTorch forward_vertices matches reference (with pose correctives)."""
    from body_models.mhr.torch import MHR

    model = MHR(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        verts = model.forward_vertices(
            shape=torch.tensor(inputs["shape"])[None],
            pose=torch.tensor(inputs["pose"])[None],
            expression=torch.tensor(inputs["expression"])[None],
        )

    # Model outputs meters, reference is in cm
    verts_cm = verts[0].numpy() * 100
    np.testing.assert_allclose(verts_cm, ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_numpy(idx: int) -> None:
    """Test NumPy forward_vertices matches reference (with pose correctives)."""
    from body_models.mhr.numpy import MHR

    model = MHR(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=inputs["shape"][None],
        pose=inputs["pose"][None],
        expression=inputs["expression"][None],
    )

    # Model outputs meters, reference is in cm
    verts_cm = verts[0] * 100
    np.testing.assert_allclose(verts_cm, ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_jax(idx: int) -> None:
    """Test JAX forward_vertices matches reference (with pose correctives)."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.mhr.jax import MHR

    model = MHR(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=jnp.array(inputs["shape"])[None],
        pose=jnp.array(inputs["pose"])[None],
        expression=jnp.array(inputs["expression"])[None],
    )

    # Model outputs meters, reference is in cm
    verts_cm = np.asarray(verts[0]) * 100
    np.testing.assert_allclose(verts_cm, ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_torch(idx: int) -> None:
    """Test PyTorch forward_skeleton matches reference."""
    from body_models.mhr import to_native_outputs
    from body_models.mhr.torch import MHR

    model = MHR(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        verts = model.forward_vertices(
            shape=torch.tensor(inputs["shape"])[None],
            pose=torch.tensor(inputs["pose"])[None],
            expression=torch.tensor(inputs["expression"])[None],
        )
        transforms = model.forward_skeleton(
            shape=torch.tensor(inputs["shape"])[None],
            pose=torch.tensor(inputs["pose"])[None],
            expression=torch.tensor(inputs["expression"])[None],
        )

    result = to_native_outputs(verts, transforms)
    _assert_skeleton_close(result["joints"][0], torch.from_numpy(ref["skeleton"]), rtol=RTOL, atol=ATOL)


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create MHR model in float64 for gradient checking."""
    from body_models.mhr.torch import MHR

    model = MHR(model_path=MODEL_PATH)
    return model.to(torch.float64).eval()


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


# ============================================================================
# Feature tests
# ============================================================================


def test_forward_accelerator_optional_defaults() -> None:
    """Test accelerator forward_* with omitted optional params stays on-device."""
    from body_models.mhr.torch import MHR

    device = get_accelerator_device()
    if device is None:
        pytest.skip("No accelerator available (cuda or mps)")

    if not MODEL_PATH.exists():
        pytest.skip(f"MHR model not found at {MODEL_PATH}")

    model = MHR(model_path=MODEL_PATH).to(device)
    B = 2
    params = model.get_rest_pose(batch_size=B)
    params["pose"] = torch.randn(B, model.pose_dim, device=device, dtype=torch.float32) * 0.05
    params.pop("expression")

    with torch.no_grad():
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)

    assert verts.device.type == device.type
    assert skel.device.type == device.type


def test_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    from body_models.mhr.torch import MHR

    model_orig = MHR(model_path=MODEL_PATH, simplify=1.0)
    model_2x = MHR(model_path=MODEL_PATH, simplify=2.0)
    model_4x = MHR(model_path=MODEL_PATH, simplify=4.0)

    # Check vertex/face counts are reduced
    assert model_2x.num_vertices < model_orig.num_vertices
    assert model_4x.num_vertices < model_2x.num_vertices
    assert model_2x.faces.shape[0] < model_orig.faces.shape[0]
    assert model_4x.faces.shape[0] < model_2x.faces.shape[0]

    # Check approximate ratios (within 10% tolerance)
    assert abs(model_2x.faces.shape[0] / model_orig.faces.shape[0] - 0.5) < 0.1
    assert abs(model_4x.faces.shape[0] / model_orig.faces.shape[0] - 0.25) < 0.1

    # Test forward pass works (with pose correctives)
    params = model_2x.get_rest_pose(batch_size=2)
    params["pose"] = torch.randn_like(params["pose"]) * 0.1  # Non-trivial pose
    verts = model_2x.forward_vertices(**params)
    skel = model_2x.forward_skeleton(**params)

    assert verts.shape == (2, model_2x.num_vertices, 3)
    assert skel.shape == (2, model_2x.num_joints, 4, 4)

    # Skeleton should be identical (computed from joint data, not vertices)
    params_orig = model_orig.get_rest_pose(batch_size=1)
    skel_orig = model_orig.forward_skeleton(**params_orig)
    skel_2x = model_2x.forward_skeleton(**params_orig)
    assert (skel_orig - skel_2x).abs().max() < 1e-6
