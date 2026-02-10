"""Tests for the ANNY body model.

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

ASSET_DIR = Path(__file__).parent / "assets" / "anny"
MODEL_PATH = ASSET_DIR / "model"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5
RTOL, ATOL = 1e-4, 1e-4

if not MODEL_PATH.exists():
    pytest.skip(f"ANNY model not found at {MODEL_PATH}", allow_module_level=True)


# ============================================================================
# Test data loading
# ============================================================================


def load_test_case(idx: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load inputs and reference outputs for a test case."""
    data = json.loads((INPUTS_DIR / f"{idx}.json").read_text())
    phenotype = data["phenotype"]
    inputs = {
        "gender": phenotype["gender"],
        "age": phenotype["age"],
        "muscle": phenotype["muscle"],
        "weight": phenotype["weight"],
        "height": phenotype["height"],
        "proportions": phenotype["proportions"],
        "pose_4x4": np.array(data["pose"], dtype=np.float32),  # [J, 4, 4]
    }
    outputs = {
        "vertices": np.load(OUTPUTS_DIR / str(idx) / "vertices.npy"),
        "bone_poses": np.load(OUTPUTS_DIR / str(idx) / "bone_poses.npy"),
    }
    return inputs, outputs


# ============================================================================
# Numerical precision tests (all backends)
# ============================================================================


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_torch(idx: int) -> None:
    """Test PyTorch forward_vertices matches reference."""
    from body_models.anny import from_native_args, to_native_outputs
    from body_models.anny.torch import ANNY

    model = ANNY(model_path=MODEL_PATH)
    model.eval()
    inputs, ref = load_test_case(idx)

    # Convert pose from 4x4 to axis-angle
    pose_4x4 = torch.tensor(inputs["pose_4x4"])[None]
    pose_args = from_native_args(pose_4x4)

    with torch.no_grad():
        verts = model.forward_vertices(
            gender=torch.tensor([inputs["gender"]], dtype=torch.float32),
            age=torch.tensor([inputs["age"]], dtype=torch.float32),
            muscle=torch.tensor([inputs["muscle"]], dtype=torch.float32),
            weight=torch.tensor([inputs["weight"]], dtype=torch.float32),
            height=torch.tensor([inputs["height"]], dtype=torch.float32),
            proportions=torch.tensor([inputs["proportions"]], dtype=torch.float32),
            **pose_args,
        )
        transforms = model.forward_skeleton(
            gender=torch.tensor([inputs["gender"]], dtype=torch.float32),
            age=torch.tensor([inputs["age"]], dtype=torch.float32),
            muscle=torch.tensor([inputs["muscle"]], dtype=torch.float32),
            weight=torch.tensor([inputs["weight"]], dtype=torch.float32),
            height=torch.tensor([inputs["height"]], dtype=torch.float32),
            proportions=torch.tensor([inputs["proportions"]], dtype=torch.float32),
            **pose_args,
        )

    # Convert to native outputs (Z-up) for comparison
    result = to_native_outputs(verts, transforms)

    np.testing.assert_allclose(result["vertices"][0].numpy(), ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result["bone_poses"][0].numpy(), ref["bone_poses"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_numpy(idx: int) -> None:
    """Test NumPy forward_vertices matches reference."""
    from body_models.anny.numpy import ANNY, from_native_args, to_native_outputs

    model = ANNY(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    # Convert pose from 4x4 to axis-angle
    pose_4x4 = inputs["pose_4x4"][None]  # [1, J, 4, 4]
    pose_args = from_native_args(pose_4x4)

    verts = model.forward_vertices(
        gender=np.array([inputs["gender"]], dtype=np.float32),
        age=np.array([inputs["age"]], dtype=np.float32),
        muscle=np.array([inputs["muscle"]], dtype=np.float32),
        weight=np.array([inputs["weight"]], dtype=np.float32),
        height=np.array([inputs["height"]], dtype=np.float32),
        proportions=np.array([inputs["proportions"]], dtype=np.float32),
        **pose_args,
    )
    transforms = model.forward_skeleton(
        gender=np.array([inputs["gender"]], dtype=np.float32),
        age=np.array([inputs["age"]], dtype=np.float32),
        muscle=np.array([inputs["muscle"]], dtype=np.float32),
        weight=np.array([inputs["weight"]], dtype=np.float32),
        height=np.array([inputs["height"]], dtype=np.float32),
        proportions=np.array([inputs["proportions"]], dtype=np.float32),
        **pose_args,
    )

    # Convert to native outputs (Z-up) for comparison
    result = to_native_outputs(verts, transforms)

    np.testing.assert_allclose(result["vertices"][0], ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result["bone_poses"][0], ref["bone_poses"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("idx", range(NUM_CASES))
def test_forward_vertices_jax(idx: int) -> None:
    """Test JAX forward_vertices matches reference."""
    jnp = pytest.importorskip("jax.numpy")
    from body_models.anny.jax import ANNY, from_native_args, to_native_outputs

    model = ANNY(model_path=MODEL_PATH)
    inputs, ref = load_test_case(idx)

    # Convert pose from 4x4 to axis-angle
    pose_4x4 = jnp.array(inputs["pose_4x4"])[None]  # [1, J, 4, 4]
    pose_args = from_native_args(pose_4x4)

    verts = model.forward_vertices(
        gender=jnp.array([inputs["gender"]], dtype=jnp.float32),
        age=jnp.array([inputs["age"]], dtype=jnp.float32),
        muscle=jnp.array([inputs["muscle"]], dtype=jnp.float32),
        weight=jnp.array([inputs["weight"]], dtype=jnp.float32),
        height=jnp.array([inputs["height"]], dtype=jnp.float32),
        proportions=jnp.array([inputs["proportions"]], dtype=jnp.float32),
        **pose_args,
    )
    transforms = model.forward_skeleton(
        gender=jnp.array([inputs["gender"]], dtype=jnp.float32),
        age=jnp.array([inputs["age"]], dtype=jnp.float32),
        muscle=jnp.array([inputs["muscle"]], dtype=jnp.float32),
        weight=jnp.array([inputs["weight"]], dtype=jnp.float32),
        height=jnp.array([inputs["height"]], dtype=jnp.float32),
        proportions=jnp.array([inputs["proportions"]], dtype=jnp.float32),
        **pose_args,
    )

    # Convert to native outputs (Z-up) for comparison
    result = to_native_outputs(verts, transforms)

    np.testing.assert_allclose(np.asarray(result["vertices"][0]), ref["vertices"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(result["bone_poses"][0]), ref["bone_poses"], rtol=RTOL, atol=ATOL)


# ============================================================================
# Gradient tests (torch only)
# ============================================================================


@pytest.fixture
def model_float64():
    """Create ANNY model in float64 for gradient checking."""
    from body_models.anny.torch import ANNY

    model = ANNY(model_path=MODEL_PATH)
    return model.to(torch.float64).eval()


def test_gradients_forward_vertices(model_float64) -> None:
    """Test gradients flow correctly through forward_vertices.

    Note: Only pose/global params are differentiable. Phenotype params (gender, age, etc.)
    go through non-differentiable ops like searchsorted.
    """
    all_params = model_float64.get_rest_pose(batch_size=1)

    # Phenotype params are not differentiable (use discrete interpolation)
    phenotype_keys = {"gender", "age", "muscle", "weight", "height", "proportions"}
    fixed_params = {k: v.to(torch.float64) for k, v in all_params.items() if k in phenotype_keys}
    grad_params = prepare_params({k: v for k, v in all_params.items() if k not in phenotype_keys})

    inputs = tuple(grad_params.values())

    def fn(*tensors):
        kwargs = {**fixed_params, **dict(zip(grad_params.keys(), tensors))}
        return model_float64.forward_vertices(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


def test_gradients_forward_skeleton(model_float64) -> None:
    """Test gradients flow correctly through forward_skeleton.

    Note: Only pose/global params are differentiable. Phenotype params (gender, age, etc.)
    go through non-differentiable ops like searchsorted.
    """
    all_params = model_float64.get_rest_pose(batch_size=1)

    # Phenotype params are not differentiable (use discrete interpolation)
    phenotype_keys = {"gender", "age", "muscle", "weight", "height", "proportions"}
    fixed_params = {k: v.to(torch.float64) for k, v in all_params.items() if k in phenotype_keys}
    grad_params = prepare_params({k: v for k, v in all_params.items() if k not in phenotype_keys})

    inputs = tuple(grad_params.values())

    def fn(*tensors):
        kwargs = {**fixed_params, **dict(zip(grad_params.keys(), tensors))}
        return model_float64.forward_skeleton(**kwargs)

    assert sampled_gradcheck(fn, inputs, n_samples=64)


# ============================================================================
# Feature tests
# ============================================================================


def test_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    from body_models.anny.torch import ANNY

    model_orig = ANNY(model_path=MODEL_PATH, simplify=1.0)
    model_2x = ANNY(model_path=MODEL_PATH, simplify=2.0)
    model_4x = ANNY(model_path=MODEL_PATH, simplify=4.0)

    # Original uses quads (F, 4), simplified uses triangles (F, 3)
    # Original: 13710 quads = 27420 equivalent triangles
    orig_triangles = model_orig.faces.shape[0] * 2

    # Check vertex/face counts are reduced
    assert model_2x.num_vertices < model_orig.num_vertices
    assert model_4x.num_vertices < model_2x.num_vertices
    assert model_2x.faces.shape[0] < orig_triangles
    assert model_4x.faces.shape[0] < model_2x.faces.shape[0]

    # Check approximate ratios (within 10% tolerance)
    assert abs(model_2x.faces.shape[0] / orig_triangles - 0.5) < 0.1
    assert abs(model_4x.faces.shape[0] / orig_triangles - 0.25) < 0.1

    # Test forward pass works
    params = model_2x.get_rest_pose(batch_size=2)
    verts = model_2x.forward_vertices(**params)
    skel = model_2x.forward_skeleton(**params)

    assert verts.shape == (2, model_2x.num_vertices, 3)
    assert skel.shape == (2, model_2x.num_joints, 4, 4)

    # Skeleton should be identical (uses bone data, not vertex regression)
    params_orig = model_orig.get_rest_pose(batch_size=1)
    skel_orig = model_orig.forward_skeleton(**params_orig)
    skel_2x = model_2x.forward_skeleton(**params_orig)
    assert (skel_orig - skel_2x).abs().max() < 1e-6
