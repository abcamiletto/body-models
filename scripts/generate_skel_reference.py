# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.9.1",
#   "scipy==1.17.0",
#   "skel @ git+https://github.com/MarilynKeller/SKEL.git",
# ]
# ///
"""Generate reference assets from the official skel package.

Runs the official SKEL implementation to produce vertices and joints, and saves
inputs/outputs under `tests/assets/skel`.

Usage:
    uv run scripts/generate_skel_reference.py /path/to/skel_models_v1.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skel.skel_model import SKEL as OfficialSKEL

torch.manual_seed(42)

TEST_ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets" / "skel"
NUM_CASES = 5


def _prepare_input_data(batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Generate random input parameters for SKEL."""
    return {
        "betas": 0.5 * torch.randn(batch_size, 10),
        "poses": 0.2 * torch.randn(batch_size, 46),
        "trans": 0.1 * torch.randn(batch_size, 3),
    }


def run(model_path: Path, gender: str = "male"):
    # Load the official SKEL model
    model = OfficialSKEL(gender=gender, model_path=model_path)
    model.eval()

    for idx in range(NUM_CASES):
        inputs = _prepare_input_data(batch_size=1)

        with torch.no_grad():
            output = model(**inputs)

        # Official SKEL returns SKELOutput dataclass with named attributes
        skin_verts = output.skin_verts[0].cpu().numpy()
        skel_verts = output.skel_verts[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()

        # Save inputs
        input_dir = TEST_ASSETS_DIR / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "gender": gender,
            "shape": inputs["betas"][0].numpy().tolist(),
            "body_pose": inputs["poses"][0].numpy().tolist(),
            "trans": inputs["trans"][0].numpy().tolist(),
        }

        input_path = input_dir / f"{idx}.json"
        with input_path.open("w") as f:
            json.dump(input_data, f, indent=4)

        # Save outputs
        output_dir = TEST_ASSETS_DIR / "outputs" / str(idx)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "vertices.npy", skin_verts)
        np.save(output_dir / "skeleton_vertices.npy", skel_verts)
        np.save(output_dir / "joints.npy", joints)

        print(f"Saved case {idx}: inputs to {input_path}, outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SKEL reference test assets")
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to SKEL model directory (skel_models_v1.1)",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    run(args.model_path)
