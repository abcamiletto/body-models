# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.4.1",
#   "torch==2.8.0",
#   "anny==0.2.0",
#   "roma==1.5.4",
#   "pyyaml==6.0.2",
#   "pillow==11.1.0",
# ]
# ///
"""Generate reference assets from the official ANNY model.

Installs the official ANNY package, runs it directly to produce vertices and
bone poses, and saves inputs/outputs under `tests/assets/anny`.
"""

import json
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)

TEST_ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets" / "anny"


def _prepare_phenotype_data(batch_size: int) -> dict[str, torch.Tensor]:
    """Generate random phenotype parameters in [0, 1] range."""
    return {
        "gender": torch.rand(batch_size),
        "age": torch.rand(batch_size),
        "muscle": torch.rand(batch_size),
        "weight": torch.rand(batch_size),
        "height": torch.rand(batch_size),
        "proportions": torch.rand(batch_size),
    }


def _prepare_pose_data(model, batch_size: int) -> torch.Tensor:
    """Generate random pose transforms as small rotations around identity."""
    import roma

    num_bones = model.bone_count
    dtype = model.dtype
    device = model.device

    # Generate small random rotations (to stay in a reasonable pose)
    axis_angles = 0.3 * torch.randn(batch_size, num_bones, 3, dtype=dtype, device=device)
    rotations = roma.rotvec_to_rotmat(axis_angles)

    # Build homogeneous transforms (rotation only, no translation)
    transforms = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
    transforms = transforms.expand(batch_size, num_bones, 4, 4).clone()
    transforms[:, :, :3, :3] = rotations

    return transforms


def run():
    import anny

    # Create the model with minimal options for reproducibility
    print("Creating ANNY model...")
    model = anny.create_fullbody_model(
        skinning_method="lbs",  # Use LBS for deterministic results
    )
    model.eval()

    batch_size = 5
    phenotype_data = _prepare_phenotype_data(batch_size)
    pose_data = _prepare_pose_data(model, batch_size)

    print("Running model forward pass...")
    with torch.no_grad():
        outputs = model(
            pose_parameters=pose_data,
            phenotype_kwargs=phenotype_data,
        )

    vertices = outputs["vertices"]
    bone_poses = outputs["bone_poses"]

    # Save bone labels for reference
    bone_labels_path = TEST_ASSETS_DIR / "bone_labels.json"
    bone_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with bone_labels_path.open("w") as f:
        json.dump(model.bone_labels, f, indent=2)
    print(f"Saved bone labels to {bone_labels_path}")

    for i in range(batch_size):
        input_dir = TEST_ASSETS_DIR / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        output_dir = TEST_ASSETS_DIR / "outputs" / str(i)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save inputs as JSON
        input_data = {
            "phenotype": {k: v[i].item() for k, v in phenotype_data.items()},
            "pose": pose_data[i].numpy().tolist(),
        }
        input_path = input_dir / f"{i}.json"
        with input_path.open("w") as f:
            json.dump(input_data, f, indent=2)

        # Save outputs as numpy arrays
        vertices_array = vertices[i].cpu().numpy()
        bone_poses_array = bone_poses[i].cpu().numpy()

        vertices_path = output_dir / "vertices.npy"
        with vertices_path.open("wb") as f:
            np.save(f, vertices_array)

        bone_poses_path = output_dir / "bone_poses.npy"
        with bone_poses_path.open("wb") as f:
            np.save(f, bone_poses_array)

        print(f"Saved inputs to {input_path} and outputs to {output_dir}")


if __name__ == "__main__":
    run()
