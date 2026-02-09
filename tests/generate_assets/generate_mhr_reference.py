# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.4.1",
#   "torch==2.9.1",
# ]
# ///
"""Generate reference assets from the official torchscript MHR model.

Clones the original MHR repo, downloads the release torchscript model, runs it
directly to produce vertices and joints, and saves inputs/outputs under
`tests/assets/mhr`.
"""

import shutil
import numpy as np
import tempfile
import zipfile
from pathlib import Path
from urllib import request

import json
import torch

torch.manual_seed(0)

URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
TEST_ASSETS_DIR = Path(__file__).parent.parent / "tests" / "assets" / "mhr"


def _download_torchscript_model(target_dir: Path) -> Path:
    archive_path = target_dir / "assets.zip"
    print(f"Downloading MHR torchscript model to {archive_path}...")
    with request.urlopen(URL) as response, open(archive_path, "wb") as archive_file:
        shutil.copyfileobj(response, archive_file)

    print("Extracting model...")
    with zipfile.ZipFile(archive_path) as zip_file:
        return Path(zip_file.extract("assets/mhr_model.pt", path=target_dir))


def _prepare_input_data(batch_size: int):
    identity_coeffs = 0.8 * torch.randn(batch_size, 45).cpu()
    model_parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5).cpu()
    face_expr_coeffs = 0.3 * torch.randn(batch_size, 72).cpu()
    return identity_coeffs, model_parameters, face_expr_coeffs


def run():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = _download_torchscript_model(Path(tmp_dir))
        scripted_model = torch.jit.load(model_path)

    batch_size = 5
    id_coeffs, model_params, face_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        vertices, skeleton = scripted_model(id_coeffs, model_params, face_coeffs)

    for i in range(batch_size):
        input_dir = TEST_ASSETS_DIR / "inputs" / str(i)
        input_dir.mkdir(parents=True, exist_ok=True)

        output_dir = TEST_ASSETS_DIR / "outputs" / str(i)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_data = {
            "shape": id_coeffs[i].numpy().tolist(),
            "expression": face_coeffs[i].numpy().tolist(),
            "pose": model_params[i].numpy().tolist(),
        }
        input_path = input_dir / "inputs.json"
        with input_path.open("w") as f:
            json.dump(input_data, f, indent=4)

        vertex_array = vertices[i].cpu().numpy()
        skeleton_array = skeleton[i].cpu().numpy()

        vertex_path = output_dir / f"{i}" / "vertices.npy"
        vertex_path.parent.mkdir(parents=True, exist_ok=True)
        with vertex_path.open("wb") as f:
            np.save(f, vertex_array)

        skeleton_path = output_dir / f"{i}" / "skeleton.npy"
        with skeleton_path.open("wb") as f:
            np.save(f, skeleton_array)

        print(f"Saved inputs to {input_dir} and outputs to {output_dir}.")


if __name__ == "__main__":
    run()
