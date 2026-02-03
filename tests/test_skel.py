from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from body_models.skel import SKEL, from_native_args, to_native_outputs

ASSET_DIR = Path(__file__).parent / "assets" / "skel"
INPUTS_DIR = ASSET_DIR / "inputs"
OUTPUTS_DIR = ASSET_DIR / "outputs"
NUM_CASES = 5


def _load_inputs(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    """Load inputs from JSON and convert to tensors matching our SKEL interface."""
    data = json.loads(path.read_text())

    gender = data["gender"]
    shape = torch.as_tensor(data["shape"], dtype=torch.float32).unsqueeze(0)
    body_pose = torch.as_tensor(data["body_pose"], dtype=torch.float32).unsqueeze(0)
    trans = torch.as_tensor(data["trans"], dtype=torch.float32).unsqueeze(0)

    return gender, {
        "shape": shape,
        "body_pose": body_pose,
        "pelvis_translation": trans,
    }


def _load_outputs(path: Path) -> dict[str, torch.Tensor]:
    """Load reference outputs from numpy files."""
    vertices = torch.from_numpy(np.load(path / "vertices.npy"))
    skeleton_vertices = torch.from_numpy(np.load(path / "skeleton_vertices.npy"))
    joints = torch.from_numpy(np.load(path / "joints.npy"))
    return {
        "vertices": vertices,
        "skeleton_vertices": skeleton_vertices,
        "joints": joints,
    }


def test_skel_matches_reference() -> None:
    """Test that our SKEL implementation matches the official skel package."""
    models: dict[str, SKEL] = {}

    for idx in range(NUM_CASES):
        input_path = INPUTS_DIR / f"{idx}.json"
        gender, inputs = _load_inputs(input_path)
        outputs = _load_outputs(OUTPUTS_DIR / str(idx))

        if gender not in models:
            models[gender] = SKEL(gender=gender)
            models[gender].eval()
        model = models[gender]

        # Convert native args to API format
        args = from_native_args(**inputs)

        with torch.no_grad():
            verts = model.forward_vertices(**args)  # type: ignore[arg-type]
            transforms = model.forward_skeleton(**args)  # type: ignore[arg-type]
            skel_mesh = model.forward_skeleton_mesh(**args)  # type: ignore[arg-type]

        # Convert to native outputs (no feet offset)
        result = to_native_outputs(verts, transforms, skel_mesh, model._feet_offset)

        ref_vertices = outputs["vertices"]
        ref_skel_vertices = outputs["skeleton_vertices"]
        ref_joints = outputs["joints"]

        torch.testing.assert_close(result["vertices"][0], ref_vertices, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result["skeleton_vertices"][0], ref_skel_vertices, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result["joints"][0], ref_joints, rtol=1e-4, atol=1e-4)
        print(f"Test case {idx} passed.")


def test_skel_simplify() -> None:
    """Test mesh simplification reduces vertex/face count and forward pass works."""
    model_orig = SKEL("male", simplify=1.0)
    model_2x = SKEL("male", simplify=2.0)
    model_4x = SKEL("male", simplify=4.0)

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
