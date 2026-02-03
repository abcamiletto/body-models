from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_models.anny import ANNY, from_native_args, to_native_outputs

ASSET_DIR = Path(__file__).parent / "assets" / "anny"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5


def _load_inputs(path: Path) -> dict[str, torch.Tensor]:
    data = json.loads(path.read_text())

    phenotype = data["phenotype"]
    pose = torch.as_tensor(np.array(data["pose"], dtype=np.float32)).unsqueeze(0)

    return {
        "gender": torch.tensor([phenotype["gender"]]),
        "age": torch.tensor([phenotype["age"]]),
        "muscle": torch.tensor([phenotype["muscle"]]),
        "weight": torch.tensor([phenotype["weight"]]),
        "height": torch.tensor([phenotype["height"]]),
        "proportions": torch.tensor([phenotype["proportions"]]),
        "pose": pose,
    }


def _load_outputs(path: Path) -> dict[str, torch.Tensor]:
    vertices_path = path / "vertices.npy"
    bone_poses_path = path / "bone_poses.npy"

    vertices = torch.from_numpy(np.load(vertices_path))
    bone_poses = torch.from_numpy(np.load(bone_poses_path))

    return {"vertices": vertices, "bone_poses": bone_poses}


def test_anny_matches_reference() -> None:
    model = ANNY()
    model.eval()

    for idx in range(NUM_CASES):
        input_path = INPUTS_DIR / f"{idx}.json"
        inputs = _load_inputs(input_path)
        outputs = _load_outputs(OUTPUTS_DIR / str(idx))

        # Convert native args (4x4 transforms) to API format (axis-angle)
        pose_args = from_native_args(inputs["pose"])

        with torch.no_grad():
            verts = model.forward_vertices(
                gender=inputs["gender"],
                age=inputs["age"],
                muscle=inputs["muscle"],
                weight=inputs["weight"],
                height=inputs["height"],
                proportions=inputs["proportions"],
                **pose_args,
            )
            transforms = model.forward_skeleton(
                gender=inputs["gender"],
                age=inputs["age"],
                muscle=inputs["muscle"],
                weight=inputs["weight"],
                height=inputs["height"],
                proportions=inputs["proportions"],
                **pose_args,
            )

        # Convert to native outputs (Z-up)
        result = to_native_outputs(verts, transforms)

        ref_vertices = outputs["vertices"]
        ref_bone_poses = outputs["bone_poses"]

        torch.testing.assert_close(result["vertices"][0], ref_vertices, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result["bone_poses"][0], ref_bone_poses, rtol=1e-4, atol=1e-4)
        print(f"Test case {idx} passed.")


def test_anny_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    model_orig = ANNY(simplify=1.0)
    model_2x = ANNY(simplify=2.0)
    model_4x = ANNY(simplify=4.0)

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
