"""Tests for the SMPL-X body model.

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

from gradient_utils import prepare_params, sampled_gradcheck

ASSET_DIR = Path(__file__).parent / "assets" / "smplx"
MODEL_PATH = ASSET_DIR / "model" / "SMPLX_NEUTRAL.npz"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5
RTOL, ATOL = 1e-4, 1e-4

if not MODEL_PATH.exists():
    pytest.skip(f"SMPLX model not found at {MODEL_PATH}", allow_module_level=True)


# ============================================================================
# Test data loading
# ============================================================================


def load_test_case(idx: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load inputs and reference outputs for a test case."""
    data = json.loads((INPUTS_DIR / f"{idx}.json").read_text())
    inputs = {
        "shape": np.array(data["shape"], dtype=np.float32),
        "expression": np.array(data["expression"], dtype=np.float32),
        "body_pose": np.array(data["body_pose"], dtype=np.float32).reshape(21, 3),
        "hand_pose": np.concatenate([data["left_hand_pose"], data["right_hand_pose"]], axis=-1)
        .astype(np.float32)
        .reshape(30, 3),
        "head_pose": np.concatenate([data["jaw_pose"], data["leye_pose"], data["reye_pose"]], axis=-1)
        .astype(np.float32)
        .reshape(3, 3),
        "pelvis_rotation": np.array(data["global_orient"], dtype=np.float32),
        "global_translation": np.array(data["transl"], dtype=np.float32),
    }
    outputs = {
        "vertices": np.load(OUTPUTS_DIR / str(idx) / "vertices.npy"),
        "joints": np.load(OUTPUTS_DIR / str(idx) / "joints.npy"),
    }
    return inputs, outputs


# ============================================================================
# Numerical precision tests (all backends)
# ============================================================================


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_torch(idx: int) -> None:
    """Test PyTorch forward_vertices matches reference."""
    from body_models.smplx.torch import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        verts = model.forward_vertices(
            shape=torch.as_tensor(inputs["shape"])[None],
            expression=torch.as_tensor(inputs["expression"])[None],
            body_pose=torch.as_tensor(inputs["body_pose"])[None],
            hand_pose=torch.as_tensor(inputs["hand_pose"])[None],
            head_pose=torch.as_tensor(inputs["head_pose"])[None],
            pelvis_rotation=torch.as_tensor(inputs["pelvis_rotation"])[None],
            global_translation=torch.as_tensor(inputs["global_translation"])[None],
        )

    np.testing.assert_allclose(verts[0].numpy(), ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_numpy(idx: int) -> None:
    """Test NumPy forward_vertices matches reference."""
    from body_models.smplx.numpy import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=inputs["shape"][None],
        expression=inputs["expression"][None],
        body_pose=inputs["body_pose"][None],
        hand_pose=inputs["hand_pose"][None],
        head_pose=inputs["head_pose"][None],
        pelvis_rotation=inputs["pelvis_rotation"][None],
        global_translation=inputs["global_translation"][None],
    )

    np.testing.assert_allclose(verts[0], ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_jax(idx: int) -> None:
    """Test JAX forward_vertices matches reference."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.smplx.jax import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=jnp.array(inputs["shape"])[None],
        expression=jnp.array(inputs["expression"])[None],
        body_pose=jnp.array(inputs["body_pose"])[None],
        hand_pose=jnp.array(inputs["hand_pose"])[None],
        head_pose=jnp.array(inputs["head_pose"])[None],
        pelvis_rotation=jnp.array(inputs["pelvis_rotation"])[None],
        global_translation=jnp.array(inputs["global_translation"])[None],
    )

    np.testing.assert_allclose(np.asarray(verts[0]), ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_torch(idx: int) -> None:
    """Test PyTorch forward_skeleton matches reference joint positions."""
    from body_models.smplx.torch import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        transforms = model.forward_skeleton(
            shape=torch.as_tensor(inputs["shape"])[None],
            expression=torch.as_tensor(inputs["expression"])[None],
            body_pose=torch.as_tensor(inputs["body_pose"])[None],
            hand_pose=torch.as_tensor(inputs["hand_pose"])[None],
            head_pose=torch.as_tensor(inputs["head_pose"])[None],
            pelvis_rotation=torch.as_tensor(inputs["pelvis_rotation"])[None],
            global_translation=torch.as_tensor(inputs["global_translation"])[None],
        )

    joints = transforms[0, :, :3, 3].numpy()
    num_joints = min(joints.shape[0], ref["joints"].shape[0])
    np.testing.assert_allclose(joints[:num_joints], ref["joints"][:num_joints], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_numpy(idx: int) -> None:
    """Test NumPy forward_skeleton matches reference joint positions."""
    from body_models.smplx.numpy import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    transforms = model.forward_skeleton(
        shape=inputs["shape"][None],
        expression=inputs["expression"][None],
        body_pose=inputs["body_pose"][None],
        hand_pose=inputs["hand_pose"][None],
        head_pose=inputs["head_pose"][None],
        pelvis_rotation=inputs["pelvis_rotation"][None],
        global_translation=inputs["global_translation"][None],
    )

    joints = transforms[0, :, :3, 3]
    num_joints = min(joints.shape[0], ref["joints"].shape[0])
    np.testing.assert_allclose(joints[:num_joints], ref["joints"][:num_joints], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_jax(idx: int) -> None:
    """Test JAX forward_skeleton matches reference joint positions."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.smplx.jax import SMPLX

    model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=False, ground_plane=False)
    inputs, ref = load_test_case(idx)

    transforms = model.forward_skeleton(
        shape=jnp.array(inputs["shape"])[None],
        expression=jnp.array(inputs["expression"])[None],
        body_pose=jnp.array(inputs["body_pose"])[None],
        hand_pose=jnp.array(inputs["hand_pose"])[None],
        head_pose=jnp.array(inputs["head_pose"])[None],
        pelvis_rotation=jnp.array(inputs["pelvis_rotation"])[None],
        global_translation=jnp.array(inputs["global_translation"])[None],
    )

    joints = np.asarray(transforms[0, :, :3, 3])
    num_joints = min(joints.shape[0], ref["joints"].shape[0])
    np.testing.assert_allclose(joints[:num_joints], ref["joints"][:num_joints], rtol=RTOL, atol=ATOL)


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create SMPLX model in float64 for gradient checking."""
    from body_models.smplx.torch import SMPLX

    model = SMPLX(model_path=MODEL_PATH)
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


def test_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    from body_models.smplx.torch import SMPLX

    model_orig = SMPLX(model_path=MODEL_PATH, simplify=1.0)
    model_2x = SMPLX(model_path=MODEL_PATH, simplify=2.0)
    model_4x = SMPLX(model_path=MODEL_PATH, simplify=4.0)

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
    assert skel.shape == (2, 55, 4, 4)

    # Skeleton should be identical (uses full-resolution mesh internally)
    params_orig = model_orig.get_rest_pose(batch_size=1)
    params_2x = model_2x.get_rest_pose(batch_size=1)
    joints_orig = model_orig.forward_skeleton(**params_orig)[0, :, :3, 3]
    joints_2x = model_2x.forward_skeleton(**params_2x)[0, :, :3, 3]
    torch.testing.assert_close(joints_orig, joints_2x, rtol=1e-5, atol=1e-5)
