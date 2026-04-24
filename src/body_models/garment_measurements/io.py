"""I/O utilities for the GarmentMeasurements PCA body model."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from .. import config
from ..utils import download_and_extract, get_cache_dir

GARMENT_MEASUREMENTS_URL = "https://github.com/mbotsch/GarmentMeasurements/archive/refs/heads/main.zip"
JOINT_NAMES = ["root"]
PARENTS = [-1]


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve a GarmentMeasurements data directory, downloading if needed."""
    if model_path is None:
        model_path = config.get_model_path("garment-measurements")

    if model_path is not None:
        model_path = Path(model_path)
        if _find_pca_files(model_path) is not None:
            return model_path
        raise FileNotFoundError(
            f"GarmentMeasurements model path {model_path} does not contain pca/point.pca and pca/mean.obj"
        )

    cache_path = get_cache_dir() / "garment_measurements"
    if _find_pca_files(cache_path) is not None:
        return cache_path
    return download_model()


def download_model() -> Path:
    """Download upstream GarmentMeasurements data assets."""
    cache_dir = get_cache_dir() / "garment_measurements"
    print(f"Downloading GarmentMeasurements model to {cache_dir}...")
    download_and_extract(
        url=GARMENT_MEASUREMENTS_URL,
        dest=cache_dir,
        extract_subdir="GarmentMeasurements-main/data/",
    )
    print("Done")
    return cache_dir


def load_model_data(model_path: Path | str | None = None, dtype: Any = np.float32) -> dict[str, Any]:
    """Load PCA data and mesh topology as NumPy arrays."""
    resolved_path = get_model_path(model_path)
    files = _find_pca_files(resolved_path)
    if files is None:
        raise FileNotFoundError(f"Missing GarmentMeasurements PCA files under {resolved_path}")

    pca_path, obj_path = files
    mean_vertices, components, eigenvalues = load_pca(pca_path, dtype=dtype)
    obj_vertices, faces = load_obj_mesh(obj_path, dtype=dtype)

    if obj_vertices.shape != mean_vertices.shape:
        raise ValueError(
            f"mean.obj vertices {obj_vertices.shape} do not match point.pca mean vertices {mean_vertices.shape}"
        )

    return {
        "mean_vertices": mean_vertices,
        "components": components,
        "eigenvalues": eigenvalues,
        "faces": faces,
        "joint_names": JOINT_NAMES,
        "parents": PARENTS,
    }


def load_pca(
    pca_path: Path | str,
    dtype: Any = np.float32,
) -> tuple[Float[np.ndarray, "V 3"], Float[np.ndarray, "V 3 C"], Float[np.ndarray, "C"]]:
    """Load the upstream binary PCA file.

    The file is written by Eigen in column-major order as:
    uint32 dimension, uint32 components, matrix[dimension, components],
    mean[dimension], eigenvalues[components].
    """
    data = Path(pca_path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"Invalid PCA file: {pca_path}")

    dimension, num_components = struct.unpack_from("<II", data, 0)
    if dimension % 3 != 0:
        raise ValueError(f"PCA dimension must be divisible by 3, got {dimension}")

    matrix_count = dimension * num_components
    expected_size = 8 + 8 * (matrix_count + dimension + num_components)
    if len(data) != expected_size:
        raise ValueError(f"Invalid PCA file size for {pca_path}: expected {expected_size}, got {len(data)}")

    offset = 8
    matrix = np.frombuffer(data, dtype="<f8", count=matrix_count, offset=offset).reshape(
        (dimension, num_components), order="F"
    )
    offset += matrix_count * 8
    mean = np.frombuffer(data, dtype="<f8", count=dimension, offset=offset)
    offset += dimension * 8
    eigenvalues = np.frombuffer(data, dtype="<f8", count=num_components, offset=offset)

    num_vertices = dimension // 3
    return (
        mean.reshape(num_vertices, 3).astype(dtype),
        matrix.reshape(num_vertices, 3, num_components).astype(dtype),
        eigenvalues.astype(dtype),
    )


def load_obj_mesh(
    obj_path: Path | str,
    dtype: Any = np.float32,
) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F _"]]:
    """Load vertices and polygon faces from the upstream OBJ mesh."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for line in Path(obj_path).read_text().splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith("f "):
            indices = [_parse_obj_index(token) for token in line.split()[1:]]
            if len(indices) >= 3:
                faces.append(indices)

    if not vertices:
        raise ValueError(f"OBJ file has no vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ file has no faces: {obj_path}")

    face_widths = {len(face) for face in faces}
    if len(face_widths) != 1:
        raise ValueError(f"Mixed face sizes are not supported in {obj_path}: {sorted(face_widths)}")

    return np.asarray(vertices, dtype=dtype), np.asarray(faces, dtype=np.int64)


def _parse_obj_index(token: str) -> int:
    index = int(token.split("/", 1)[0])
    if index <= 0:
        raise ValueError("Only positive OBJ indices are supported")
    return index - 1


def _find_pca_files(model_path: Path) -> tuple[Path, Path] | None:
    for base in (model_path, model_path / "data"):
        pca_path = base / "pca" / "point.pca"
        obj_path = base / "pca" / "mean.obj"
        if pca_path.is_file() and obj_path.is_file():
            return pca_path, obj_path
    return None


__all__ = [
    "GARMENT_MEASUREMENTS_URL",
    "JOINT_NAMES",
    "PARENTS",
    "download_model",
    "get_model_path",
    "load_model_data",
    "load_obj_mesh",
    "load_pca",
]
