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
PREPROCESSED_FILENAME = "garment_measurements.npz"


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve a GarmentMeasurements data directory, downloading if needed."""
    if model_path is None:
        model_path = config.get_model_path("garment-measurements")

    if model_path is not None:
        model_path = Path(model_path)
        if _find_preprocessed_file(model_path) is not None or _find_pca_files(model_path) is not None:
            return model_path
        raise FileNotFoundError(
            f"GarmentMeasurements model path {model_path} does not contain {PREPROCESSED_FILENAME} "
            "or pca/point.pca and pca/mean.obj"
        )

    cache_path = get_cache_dir() / "garment_measurements"
    if _find_preprocessed_file(cache_path) is not None or _find_pca_files(cache_path) is not None:
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


def load_model_data(
    model_path: Path | str | None = None,
    dtype: Any = np.float32,
    *,
    require_skeleton: bool = True,
) -> dict[str, Any]:
    """Load preprocessed model data as NumPy arrays.

    The runtime model consumes a dependency-free ``garment_measurements.npz`` generated
    from upstream ``male.fbx``. Set ``require_skeleton=False`` only for tests or tools
    that intentionally operate on the upstream PCA files without articulation.
    """
    resolved_path = get_model_path(model_path)
    model_file = _find_preprocessed_file(resolved_path)
    if model_file is not None:
        return load_preprocessed_model(model_file, dtype=dtype)

    if require_skeleton:
        raise FileNotFoundError(
            f"Missing {PREPROCESSED_FILENAME} under {resolved_path}. Generate it from upstream male.fbx with "
            "tests/generate_assets/generate_garment_measurements_reference.py and place it in the private "
            "garment_measurements/model asset directory."
        )

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
    }


def load_preprocessed_model(model_path: Path | str, dtype: Any = np.float32) -> dict[str, Any]:
    """Load a preprocessed dependency-free GarmentMeasurements ``.npz`` asset."""
    path = Path(model_path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "mean_vertices",
            "components",
            "eigenvalues",
            "faces",
            "joint_names",
            "parents",
            "bind_quats",
            "skin_weights",
            "mvc_weights",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")

        result = {
            "mean_vertices": np.asarray(data["mean_vertices"], dtype=dtype),
            "components": np.asarray(data["components"], dtype=dtype),
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=dtype),
            "faces": np.asarray(data["faces"], dtype=np.int64),
            "joint_names": [str(name) for name in data["joint_names"].tolist()],
            "parents": np.asarray(data["parents"], dtype=np.int64),
            "bind_quats": np.asarray(data["bind_quats"], dtype=dtype),
            "skin_weights": np.asarray(data["skin_weights"], dtype=dtype),
            "mvc_weights": np.asarray(data["mvc_weights"], dtype=dtype),
        }

    _validate_preprocessed_model(path, result)
    return result


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


def _find_preprocessed_file(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.name == PREPROCESSED_FILENAME:
        return model_path
    for base in (model_path, model_path / "data"):
        path = base / PREPROCESSED_FILENAME
        if path.is_file():
            return path
    return None


def _validate_preprocessed_model(path: Path, data: dict[str, Any]) -> None:
    num_vertices = data["mean_vertices"].shape[0]
    num_joints = len(data["joint_names"])
    num_components = data["eigenvalues"].shape[0]

    expected = {
        "mean_vertices": (num_vertices, 3),
        "components": (num_vertices, 3, num_components),
        "parents": (num_joints,),
        "bind_quats": (num_joints, 4),
        "skin_weights": (num_vertices, num_joints),
        "mvc_weights": (num_vertices, num_joints),
    }
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"{path} array {key} has shape {data[key].shape}, expected {shape}")

    if data["parents"][0] != -1:
        raise ValueError(f"{path} parents[0] must be -1")


__all__ = [
    "GARMENT_MEASUREMENTS_URL",
    "PREPROCESSED_FILENAME",
    "download_model",
    "get_model_path",
    "load_model_data",
    "load_obj_mesh",
    "load_pca",
    "load_preprocessed_model",
]
