"""Tests for the SMPL body model.

- Numerical precision: parametrized across torch/numpy/jax backends
- Gradient correctness: torch backend with gradcheck
- Feature tests: mesh simplification, etc.
"""

import json
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from nanomanifold import SO3

from accelerator_utils import get_accelerator_device
from gradient_utils import prepare_params, sampled_gradcheck

ASSET_DIR = Path(__file__).parent / "assets" / "smpl"
MODEL_PATH = ASSET_DIR / "model" / "SMPL_NEUTRAL.npz"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5
RTOL, ATOL = 1e-4, 1e-4
ROTATION_TYPES = ["axis_angle", "quat", "sixd", "matrix"]

if not MODEL_PATH.exists():
    pytest.skip(f"SMPL model not found at {MODEL_PATH}", allow_module_level=True)


# ============================================================================
# Test data loading
# ============================================================================


def load_test_case(idx: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load inputs and reference outputs for a test case."""
    data = json.loads((INPUTS_DIR / f"{idx}.json").read_text())
    inputs = {
        "shape": np.array(data["shape"], dtype=np.float32),
        "body_pose": np.array(data["body_pose"], dtype=np.float32).reshape(23, 3),
        "pelvis_rotation": np.array(data["global_orient"], dtype=np.float32),
        "global_translation": np.array(data["transl"], dtype=np.float32),
    }
    outputs = {
        "vertices": np.load(OUTPUTS_DIR / str(idx) / "vertices.npy"),
        "joints": np.load(OUTPUTS_DIR / str(idx) / "joints.npy"),
    }
    return inputs, outputs


def convert_rotation_inputs(inputs: dict[str, np.ndarray], rotation_type: str) -> dict[str, np.ndarray]:
    if rotation_type == "axis_angle":
        return inputs

    return {
        "shape": inputs["shape"],
        "body_pose": SO3.convert(inputs["body_pose"], src="axis_angle", dst=rotation_type, xp=np),
        "pelvis_rotation": SO3.convert(inputs["pelvis_rotation"], src="axis_angle", dst=rotation_type, xp=np),
        "global_translation": inputs["global_translation"],
    }


def _smpl_backend(backend: str):
    if backend == "jax":
        pytest.importorskip("jax.numpy")
    module = import_module(f"body_models.smpl.{backend}")
    return getattr(module, "SMPL")


def _backend_array(backend: str, value: np.ndarray):
    if backend == "torch":
        return torch.tensor(value)
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.array(value)
    return value


def _to_numpy(backend: str, value):
    if backend == "torch":
        return value.numpy()
    return np.asarray(value)


# ============================================================================
# Numerical precision tests (all backends)
# ============================================================================


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_torch(idx: int) -> None:
    """Test PyTorch forward_vertices matches reference."""
    from body_models.smpl.torch import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        verts = model.forward_vertices(
            shape=torch.tensor(inputs["shape"])[None],
            body_pose=torch.tensor(inputs["body_pose"])[None],
            pelvis_rotation=torch.tensor(inputs["pelvis_rotation"])[None],
            global_translation=torch.tensor(inputs["global_translation"])[None],
        )

    np.testing.assert_allclose(verts[0].numpy(), ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_numpy(idx: int) -> None:
    """Test NumPy forward_vertices matches reference."""
    from body_models.smpl.numpy import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=inputs["shape"][None],
        body_pose=inputs["body_pose"][None],
        pelvis_rotation=inputs["pelvis_rotation"][None],
        global_translation=inputs["global_translation"][None],
    )

    np.testing.assert_allclose(verts[0], ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_jax(idx: int) -> None:
    """Test JAX forward_vertices matches reference."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.smpl.jax import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    verts = model.forward_vertices(
        shape=jnp.array(inputs["shape"])[None],
        body_pose=jnp.array(inputs["body_pose"])[None],
        pelvis_rotation=jnp.array(inputs["pelvis_rotation"])[None],
        global_translation=jnp.array(inputs["global_translation"])[None],
    )

    np.testing.assert_allclose(np.asarray(verts[0]), ref["vertices"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_torch(idx: int) -> None:
    """Test PyTorch forward_skeleton matches reference joint positions."""
    from body_models.smpl.torch import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    with torch.no_grad():
        transforms = model.forward_skeleton(
            shape=torch.tensor(inputs["shape"])[None],
            body_pose=torch.tensor(inputs["body_pose"])[None],
            pelvis_rotation=torch.tensor(inputs["pelvis_rotation"])[None],
            global_translation=torch.tensor(inputs["global_translation"])[None],
        )

    joints = transforms[0, :, :3, 3].numpy()
    num_joints = min(joints.shape[0], ref["joints"].shape[0])
    np.testing.assert_allclose(joints[:num_joints], ref["joints"][:num_joints], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_skeleton_numpy(idx: int) -> None:
    """Test NumPy forward_skeleton matches reference joint positions."""
    from body_models.smpl.numpy import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    transforms = model.forward_skeleton(
        shape=inputs["shape"][None],
        body_pose=inputs["body_pose"][None],
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
    from body_models.smpl.jax import SMPL

    model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    inputs, ref = load_test_case(idx)

    transforms = model.forward_skeleton(
        shape=jnp.array(inputs["shape"])[None],
        body_pose=jnp.array(inputs["body_pose"])[None],
        pelvis_rotation=jnp.array(inputs["pelvis_rotation"])[None],
        global_translation=jnp.array(inputs["global_translation"])[None],
    )

    joints = np.asarray(transforms[0, :, :3, 3])
    num_joints = min(joints.shape[0], ref["joints"].shape[0])
    np.testing.assert_allclose(joints[:num_joints], ref["joints"][:num_joints], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("rotation_type", ROTATION_TYPES)
def test_rotation_types(rotation_type: str, backend: str) -> None:
    """Test SMPL matches axis-angle across rotation representations."""
    SMPL = _smpl_backend(backend)
    inputs, _ = load_test_case(0)
    native_model = SMPL(model_path=MODEL_PATH, ground_plane=False)
    rotated_model = SMPL(model_path=MODEL_PATH, ground_plane=False, rotation_type=rotation_type)

    rotated_inputs = convert_rotation_inputs(inputs, rotation_type)
    native_kwargs = {k: _backend_array(backend, v)[None] for k, v in inputs.items()}
    rotated_kwargs = {k: _backend_array(backend, v)[None] for k, v in rotated_inputs.items()}

    context = torch.no_grad() if backend == "torch" else nullcontext()
    with context:
        native_verts = native_model.forward_vertices(**native_kwargs)
        native_skeleton = native_model.forward_skeleton(**native_kwargs)
        rotated_verts = rotated_model.forward_vertices(**rotated_kwargs)
        rotated_skeleton = rotated_model.forward_skeleton(**rotated_kwargs)

    np.testing.assert_allclose(
        _to_numpy(backend, rotated_verts), _to_numpy(backend, native_verts), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        _to_numpy(backend, rotated_skeleton),
        _to_numpy(backend, native_skeleton),
        rtol=RTOL,
        atol=ATOL,
    )


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create SMPL model in float64 for gradient checking."""
    from body_models.smpl.torch import SMPL

    model = SMPL(model_path=MODEL_PATH)
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
    from body_models.smpl.torch import SMPL

    device = get_accelerator_device()
    if device is None:
        pytest.skip("No accelerator available (cuda or mps)")

    model = SMPL(model_path=MODEL_PATH, ground_plane=True).to(device)
    B = 2
    params = model.get_rest_pose(batch_size=B)
    params["body_pose"] = torch.randn(B, model.NUM_BODY_JOINTS, 3, device=device, dtype=torch.float32)
    params.pop("pelvis_rotation")

    with torch.no_grad():
        verts = model.forward_vertices(**params)
        skel = model.forward_skeleton(**params)

    assert verts.device.type == device.type
    assert skel.device.type == device.type


def test_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    from body_models.smpl.torch import SMPL

    model_orig = SMPL(model_path=MODEL_PATH, simplify=1.0)
    model_2x = SMPL(model_path=MODEL_PATH, simplify=2.0)
    model_4x = SMPL(model_path=MODEL_PATH, simplify=4.0)

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

    # Test with non-trivial pose
    torch.manual_seed(42)
    params_posed = {
        "shape": torch.randn(1, 10),
        "body_pose": torch.randn(1, 23, 3) * 0.3,
    }
    verts_orig = model_orig.forward_vertices(**params_posed)
    verts_2x = model_2x.forward_vertices(**params_posed)

    # Bounding boxes should be similar (within 5cm)
    bb_orig = verts_orig.max(dim=1).values - verts_orig.min(dim=1).values
    bb_2x = verts_2x.max(dim=1).values - verts_2x.min(dim=1).values
    assert (bb_orig - bb_2x).abs().max() < 0.05
