import json
from pathlib import Path

import numpy as np
import torch

from body_models.smpl import SMPL, from_native_args, to_native_outputs

ASSET_DIR = Path(__file__).parent / "assets" / "smpl"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5


def _load_inputs(path: Path) -> dict[str, torch.Tensor]:
    """Load inputs from JSON and convert to tensors matching our SMPL interface."""
    data = json.loads(path.read_text())

    shape = torch.as_tensor(data["shape"], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.as_tensor(data["body_pose"], dtype=torch.float32).unsqueeze(0)

    # The official smplx uses global_orient as the root/pelvis rotation
    pelvis_rotation = torch.as_tensor(data["global_orient"], dtype=torch.float32).unsqueeze(0)
    pelvis_translation = torch.as_tensor(data["transl"], dtype=torch.float32).unsqueeze(0)

    return {
        "shape": shape,
        "body_pose": body_pose,
        "pelvis_rotation": pelvis_rotation,
        "pelvis_translation": pelvis_translation,
    }


def _load_outputs(path: Path) -> dict[str, torch.Tensor]:
    """Load reference outputs from numpy files."""
    vertices = torch.from_numpy(np.load(path / "vertices.npy"))
    joints = torch.from_numpy(np.load(path / "joints.npy"))
    return {"vertices": vertices, "joints": joints}


def test_smpl_matches_reference() -> None:
    """Test that our SMPL implementation matches the official smplx package."""
    model = SMPL(ground_plane=False)  # Use native coordinates for comparison
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

        # Convert to native outputs (joint positions instead of transforms)
        result = to_native_outputs(verts, transforms)

        ref_vertices = outputs["vertices"]
        ref_joints = outputs["joints"]

        torch.testing.assert_close(result["vertices"][0], ref_vertices, rtol=1e-4, atol=1e-4)
        # Note: joint count may differ (official smplx has extra regressed joints)
        # Compare only the first NUM_JOINTS joints
        joints = result["joints"][0]
        num_joints = min(joints.shape[0], ref_joints.shape[0])
        torch.testing.assert_close(joints[:num_joints], ref_joints[:num_joints], rtol=1e-4, atol=1e-4)
        print(f"Test case {idx} passed.")


def test_smpl_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    model_orig = SMPL(simplify=1.0)
    model_2x = SMPL(simplify=2.0)
    model_4x = SMPL(simplify=4.0)

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
