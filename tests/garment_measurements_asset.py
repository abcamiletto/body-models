"""Synthetic GarmentMeasurements test asset in the upstream file layout."""

from pathlib import Path
import struct
import tempfile

import numpy as np


def get_garment_measurements_model_path() -> Path:
    """Create and return a reusable tiny GarmentMeasurements model directory."""
    model_dir = Path(tempfile.gettempdir()) / "body_models_garment_measurements_test_asset"
    pca_dir = model_dir / "pca"
    pca_path = pca_dir / "point.pca"
    obj_path = pca_dir / "mean.obj"
    if pca_path.exists() and obj_path.exists():
        return model_dir

    pca_dir.mkdir(parents=True, exist_ok=True)
    mean, components, eigenvalues, faces = synthetic_garment_measurements_data()

    matrix = components.reshape(-1, components.shape[-1]).astype(np.float64)
    payload = bytearray(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
    payload.extend(np.asfortranarray(matrix).tobytes(order="F"))
    payload.extend(mean.astype(np.float64).reshape(-1).tobytes())
    payload.extend(eigenvalues.astype(np.float64).tobytes())
    pca_path.write_bytes(payload)

    obj_lines = ["# synthetic GarmentMeasurements fixture"]
    obj_lines.extend(f"v {x} {y} {z}" for x, y, z in mean)
    obj_lines.extend("f " + " ".join(str(index + 1) for index in face) for face in faces)
    obj_path.write_text("\n".join(obj_lines) + "\n")
    return model_dir


def synthetic_garment_measurements_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    components = np.zeros((4, 3, 2), dtype=np.float32)
    components[:, :, 0] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    components[:, :, 1] = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [-0.5, -0.5, -0.5],
        ],
        dtype=np.float32,
    )
    eigenvalues = np.array([4.0, 9.0], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return mean, components, eigenvalues, faces
