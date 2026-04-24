"""I/O utilities for the GarmentMeasurements PCA body model."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..utils import download_and_extract, get_cache_dir

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

GARMENT_MEASUREMENTS_URL = "https://github.com/mbotsch/GarmentMeasurements/archive/refs/heads/main.zip"
PREPROCESSED_FILENAME = "garment_measurements.npz"
GENERATOR_PYTHON = "3.11"


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve a GarmentMeasurements data directory, downloading if needed."""
    if model_path is None:
        model_path = config.get_model_path("garment-measurements")

    if model_path is not None:
        model_path = Path(model_path)
        if _find_preprocessed_file(model_path) is not None or _find_upstream_data_dir(model_path) is not None:
            return model_path
        raise FileNotFoundError(
            f"GarmentMeasurements model path {model_path} does not contain {PREPROCESSED_FILENAME} "
            "or upstream pca/point.pca, pca/mean.obj, and template/male.fbx"
        )

    cache_path = get_cache_dir() / "garment_measurements"
    if _find_preprocessed_file(cache_path) is not None or _find_upstream_data_dir(cache_path) is not None:
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
    """Load preprocessed model data as NumPy arrays."""
    resolved_path = get_model_path(model_path)
    model_file = _find_preprocessed_file(resolved_path)
    if model_file is not None:
        return load_preprocessed_model(model_file, dtype=dtype)

    upstream_data = _find_upstream_data_dir(resolved_path)
    if upstream_data is not None:
        return load_preprocessed_model(preprocess_model(upstream_data), dtype=dtype)

    raise FileNotFoundError(
        f"Missing {PREPROCESSED_FILENAME} under {resolved_path}. Provide either a preprocessed model file "
        "or an upstream GarmentMeasurements data directory with pca/point.pca, pca/mean.obj, and template/male.fbx."
    )


def preprocess_model(upstream_data: Path | str, output_dir: Path | str | None = None) -> Path:
    """Generate ``garment_measurements.npz`` from an upstream GarmentMeasurements data directory."""
    resolved_upstream = _find_upstream_data_dir(Path(upstream_data))
    if resolved_upstream is None:
        raise FileNotFoundError(
            "Expected an upstream GarmentMeasurements data directory containing "
            "pca/point.pca, pca/mean.obj, and template/male.fbx"
        )

    output_dir = Path(output_dir) if output_dir is not None else _preprocessed_output_dir(resolved_upstream)
    output_file = output_dir / PREPROCESSED_FILENAME
    if output_file.is_file():
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    _run_asset_generator(resolved_upstream, output_dir)
    if not output_file.is_file():
        raise RuntimeError(f"GarmentMeasurements preprocessing did not produce {output_file}")
    return output_file


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


def compute_kinematic_fronts(parents: np.ndarray | list[int]) -> list[Front]:
    """Compute kinematic fronts for batched FK."""
    parents_list = parents.tolist() if isinstance(parents, np.ndarray) else list(parents)
    processed: set[int] = set()
    fronts: list[Front] = []

    while len(processed) < len(parents_list):
        joints: list[int] = []
        joint_parents: list[int] = []
        for joint_index, parent_index in enumerate(parents_list):
            if joint_index in processed:
                continue
            if parent_index < 0 or parent_index in processed:
                joints.append(joint_index)
                joint_parents.append(int(parent_index))
        if not joints:
            raise ValueError(f"Invalid GarmentMeasurements parent chain: {parents_list}")
        fronts.append((joints, joint_parents))
        processed.update(joints)

    return fronts


def _find_upstream_data_dir(model_path: Path) -> Path | None:
    for base in (model_path, model_path / "data"):
        pca_path = base / "pca" / "point.pca"
        obj_path = base / "pca" / "mean.obj"
        fbx_path = base / "template" / "male.fbx"
        if pca_path.is_file() and obj_path.is_file() and fbx_path.is_file():
            return base
    return None


def _find_preprocessed_file(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.name == PREPROCESSED_FILENAME:
        return model_path
    for base in (model_path, model_path / "data"):
        path = base / PREPROCESSED_FILENAME
        if path.is_file():
            return path
    return None


def _preprocessed_output_dir(upstream_data: Path) -> Path:
    resolved = str(upstream_data.resolve())
    digest = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    return get_cache_dir() / "garment_measurements" / "processed" / digest


def _run_asset_generator(upstream_data: Path, output_dir: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "GarmentMeasurements preprocessing requires `uv` on PATH because the FBX converter "
            "is a self-contained PEP 723 script that installs bpy outside the runtime environment."
        )

    script = Path(__file__).with_name("generate_asset.py")
    output_file = output_dir / PREPROCESSED_FILENAME
    command = [
        uv,
        "run",
        "--python",
        GENERATOR_PYTHON,
        "--no-project",
        str(script),
        str(upstream_data),
        str(output_dir),
    ]

    print("GarmentMeasurements: preprocessing upstream FBX data with bpy via uv.")
    print(f"GarmentMeasurements: generated asset will be saved to {output_file}")
    print(f"GarmentMeasurements: running {' '.join(command)}")
    subprocess.run(command, check=True)
    print(f"GarmentMeasurements: generated {output_file}")
    print(f"GarmentMeasurements: to reuse it directly, run `body-models set garment-measurements {output_dir}`")


def _validate_preprocessed_model(path: Path, data: dict[str, Any]) -> None:
    num_vertices = data["mean_vertices"].shape[0]
    num_joints = len(data["joint_names"])
    num_components = data["eigenvalues"].shape[0]

    expected = {
        "mean_vertices": (num_vertices, 3),
        "components": (num_vertices, 3, num_components),
        "faces": (data["faces"].shape[0], 3),
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
    "Front",
    "GARMENT_MEASUREMENTS_URL",
    "GENERATOR_PYTHON",
    "PREPROCESSED_FILENAME",
    "compute_kinematic_fronts",
    "download_model",
    "get_model_path",
    "load_model_data",
    "load_preprocessed_model",
    "preprocess_model",
]
