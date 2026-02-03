# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.4.1",
#   "torch==2.9.1",
#   "smplx==0.1.28",
# ]
# ///
"""Generate reference assets from the official smplx package (SMPL model).

Runs the official smplx implementation to produce vertices and joints, and saves
inputs/outputs under `tests/assets/smpl`.

Usage:
    uv run scripts/generate_smpl_reference.py /path/to/SMPL_NEUTRAL.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch

torch.manual_seed(42)

TEST_ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets" / "smpl"
NUM_CASES = 5


def _prepare_input_data(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Generate random input parameters for SMPL."""
    return {
        "betas": 0.5 * torch.randn(batch_size, 10),
        "body_pose": 0.2 * torch.randn(batch_size, 69),  # 23 joints * 3
        "global_orient": 0.2 * torch.randn(batch_size, 3),
        "transl": 0.1 * torch.randn(batch_size, 3),
    }


def run(model_path: Path):
    # Load the official SMPL model from smplx package
    model = smplx.SMPL(
        model_path=str(model_path),
        gender="neutral",
    )
    model.eval()

    for idx in range(NUM_CASES):
        inputs = _prepare_input_data(batch_size=1)

        with torch.no_grad():
            output = model(**inputs)

        vertices = output.vertices[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()

        # Save inputs
        input_dir = TEST_ASSETS_DIR / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "shape": inputs["betas"][0].numpy().tolist(),
            "body_pose": inputs["body_pose"][0].numpy().tolist(),
            "global_orient": inputs["global_orient"][0].numpy().tolist(),
            "transl": inputs["transl"][0].numpy().tolist(),
        }

        input_path = input_dir / f"{idx}.json"
        with input_path.open("w") as f:
            json.dump(input_data, f, indent=4)

        # Save outputs
        output_dir = TEST_ASSETS_DIR / "outputs" / str(idx)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "vertices.npy", vertices)
        np.save(output_dir / "joints.npy", joints)

        print(f"Saved case {idx}: inputs to {input_path}, outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SMPL reference test assets")
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to SMPL_NEUTRAL.npz model file",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    run(args.model_path)
