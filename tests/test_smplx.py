from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_models.smplx import SMPLX, from_native_args, to_native_outputs

ASSET_DIR = Path(__file__).parent / "assets" / "smplx"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5


def _load_inputs(path: Path) -> dict[str, torch.Tensor]:
    """Load inputs from JSON and convert to tensors matching our SMPLX interface."""
    data = json.loads(path.read_text())

    shape = torch.as_tensor(data["shape"], dtype=torch.float32).unsqueeze(0)
    expression = torch.as_tensor(data["expression"], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.as_tensor(data["body_pose"], dtype=torch.float32).unsqueeze(0)

    # Concatenate hand poses (left + right)
    left_hand = torch.as_tensor(data["left_hand_pose"], dtype=torch.float32)
    right_hand = torch.as_tensor(data["right_hand_pose"], dtype=torch.float32)
    hand_pose = torch.cat([left_hand, right_hand], dim=-1).unsqueeze(0)

    # Concatenate head poses (jaw + leye + reye)
    jaw = torch.as_tensor(data["jaw_pose"], dtype=torch.float32)
    leye = torch.as_tensor(data["leye_pose"], dtype=torch.float32)
    reye = torch.as_tensor(data["reye_pose"], dtype=torch.float32)
    head_pose = torch.cat([jaw, leye, reye], dim=-1).unsqueeze(0)

    # The official smplx uses global_orient as the root/pelvis rotation
    pelvis_rotation = torch.as_tensor(data["global_orient"], dtype=torch.float32).unsqueeze(0)
    pelvis_translation = torch.as_tensor(data["transl"], dtype=torch.float32).unsqueeze(0)

    return {
        "shape": shape,
        "expression": expression,
        "body_pose": body_pose,
        "hand_pose": hand_pose,
        "head_pose": head_pose,
        "pelvis_rotation": pelvis_rotation,
        "pelvis_translation": pelvis_translation,
    }


def _load_outputs(path: Path) -> dict[str, torch.Tensor]:
    """Load reference outputs from numpy files."""
    vertices = torch.from_numpy(np.load(path / "vertices.npy"))
    joints = torch.from_numpy(np.load(path / "joints.npy"))
    return {"vertices": vertices, "joints": joints}


def test_smplx_matches_reference() -> None:
    """Test that our SMPLX implementation matches the official smplx package."""
    model = SMPLX(flat_hand_mean=False, use_hand_pca=False)
    model.eval()

    for idx in range(NUM_CASES):
        input_path = INPUTS_DIR / f"{idx}.json"
        inputs = _load_inputs(input_path)
        outputs = _load_outputs(OUTPUTS_DIR / str(idx))

        # Convert native args to API format
        args = from_native_args(**inputs)

        with torch.no_grad():
            verts = model.forward_vertices(**args)  # type: ignore[arg-type]
            transforms = model.forward_skeleton(**args)  # type: ignore[arg-type]

        # Convert to native outputs (no feet offset, joint positions)
        result = to_native_outputs(verts, transforms, model.gender)

        ref_vertices = outputs["vertices"]
        ref_joints = outputs["joints"]

        torch.testing.assert_close(result["vertices"][0], ref_vertices, rtol=1e-4, atol=1e-4)
        # Note: joint count may differ (official smplx has extra regressed joints)
        # Compare only the first NUM_JOINTS joints
        joints = result["joints"][0]
        num_joints = min(joints.shape[0], ref_joints.shape[0])
        torch.testing.assert_close(joints[:num_joints], ref_joints[:num_joints], rtol=1e-4, atol=1e-4)
        print(f"Test case {idx} passed.")


def test_smplx_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    model_orig = SMPLX(simplify=1.0)
    model_2x = SMPLX(simplify=2.0)
    model_4x = SMPLX(simplify=4.0)

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
