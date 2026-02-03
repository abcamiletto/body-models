from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_models.mhr import MHR, from_native_args, to_native_outputs

ASSET_DIR = Path(__file__).parent / "assets" / "mhr"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5


def _load_inputs(path: Path) -> dict[str, torch.Tensor]:
    data = json.loads(path.read_text())

    shape = np.asarray(data["shape"], dtype=np.float32)
    expression = np.asarray(data["expression"], dtype=np.float32)
    pose = np.asarray(data["pose"], dtype=np.float32)

    return {
        "shape": torch.as_tensor(shape).unsqueeze(0),
        "expression": torch.as_tensor(expression).unsqueeze(0),
        "pose": torch.as_tensor(pose).unsqueeze(0),
    }


def _load_outputs(path: Path) -> dict[str, torch.Tensor]:
    vertices_path = path / "vertices.npy"
    joints_path = path / "skeleton.npy"

    vertices = torch.from_numpy(np.load(vertices_path))
    joints = torch.from_numpy(np.load(joints_path))

    return {"vertices": vertices, "joints": joints}


def _assert_skeleton_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Compare skeleton states, handling quaternion double cover (q ≈ -q)."""
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


def test_mhr_matches_reference() -> None:
    model = MHR()
    model.eval()

    for idx in range(NUM_CASES):
        input_path = INPUTS_DIR / f"{idx}.json"
        inputs = _load_inputs(input_path)
        outputs = _load_outputs(OUTPUTS_DIR / str(idx))

        # Convert native args (shape, expression, pose) to API format
        args = from_native_args(inputs["shape"], inputs["expression"], inputs["pose"])

        with torch.no_grad():
            verts = model.forward_vertices(**args)
            transforms = model.forward_skeleton(**args)

        # Convert to native outputs (cm units)
        result = to_native_outputs(verts, transforms)

        ref_vertices = outputs["vertices"]
        ref_joints = outputs["joints"]

        torch.testing.assert_close(result["vertices"][0], ref_vertices, rtol=1e-4, atol=1e-4)
        _assert_skeleton_close(result["joints"][0], ref_joints, rtol=1e-4, atol=1e-4)
        print(f"Test case {idx} passed.")


def test_mhr_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    model_orig = MHR(simplify=1.0)
    model_2x = MHR(simplify=2.0)
    model_4x = MHR(simplify=4.0)

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
