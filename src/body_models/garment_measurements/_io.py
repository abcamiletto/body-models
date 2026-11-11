"""I/O utilities for the GarmentMeasurements PCA body model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models import _download
from body_models._common import kinematics
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
Array = Any

PREPROCESSED_FILENAME = "garment_measurements.npz"
_FLOAT_ARRAYS = ("mean_vertices", "components", "eigenvalues", "bind_quats", "skin_weights", "mvc_weights")


@dataclass(frozen=True)
class GarmentMeasurementsAssets:
    mean_vertices: Float[Array, "V 3"]
    components: Float[Array, "V 3 C"]
    eigenvalues: Float[Array, "C"]
    faces: Int[Array, "F 3"]
    joint_names: list[str]
    kinematic_tree: kinematics.KinematicTree
    bind_quats: Float[Array, "J 4"]
    skin_weights: Float[Array, "V J"]
    compact_skinning: CompactSkinning
    mvc_weights: Float[Array, "V J"]


def validate_path(model_path: PathLike) -> Path:
    """Validate a directory containing the preprocessed ``garment_measurements.npz``."""
    model_path = Path(model_path)
    if (model_path / PREPROCESSED_FILENAME).is_file():
        return model_path
    upstream_data = _find_upstream_data_dir(model_path)
    if upstream_data is not None:
        raise FileNotFoundError(_preprocess_message(upstream_data))
    raise FileNotFoundError(
        f"GarmentMeasurements model directory {model_path} does not contain {PREPROCESSED_FILENAME}"
    )


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download the preprocessed public GarmentMeasurements assets unless they are already present."""
    return _download.download_public_assets("garment_measurements", output_dir, validate_path)


def load_model_data(model_dir: Path, dtype: Any = np.float32) -> GarmentMeasurementsAssets:
    """Load the preprocessed ``garment_measurements.npz`` from a validated model directory."""
    path = model_dir / PREPROCESSED_FILENAME
    with np.load(path, allow_pickle=False) as npz:
        data = {key: np.asarray(npz[key], dtype=dtype) for key in _FLOAT_ARRAYS}
        data["faces"] = np.asarray(npz["faces"], dtype=np.int64)
        data["parents"] = np.asarray(npz["parents"], dtype=np.int64)
        joint_names = [str(name) for name in npz["joint_names"].tolist()]
    _validate_shapes(path, data, num_joints=len(joint_names))

    skin_joint_indices, skin_joint_weights = kinematics.compute_sparse_skin_weights(data["skin_weights"])
    return GarmentMeasurementsAssets(
        mean_vertices=data["mean_vertices"],
        components=data["components"],
        eigenvalues=data["eigenvalues"],
        faces=data["faces"],
        joint_names=joint_names,
        kinematic_tree=kinematics.KinematicTree.from_parents(data["parents"]),
        bind_quats=data["bind_quats"],
        skin_weights=data["skin_weights"],
        compact_skinning=CompactSkinning(skin_joint_indices, skin_joint_weights),
        mvc_weights=data["mvc_weights"],
    )


def _find_upstream_data_dir(model_path: Path) -> Path | None:
    for base in (model_path, model_path / "data"):
        pca_path = base / "pca" / "point.pca"
        obj_path = base / "pca" / "mean.obj"
        fbx_path = base / "template" / "male.fbx"
        if pca_path.is_file() and obj_path.is_file() and fbx_path.is_file():
            return base
    return None


def _preprocess_message(upstream_data: Path) -> str:
    script = Path(__file__).with_name("_tools") / "generate_asset.py"
    return (
        f"GarmentMeasurements requires a preprocessed {PREPROCESSED_FILENAME} asset. "
        "Generate it from the upstream files with:\n"
        f"uv run --python 3.11 --no-project {script} {upstream_data} /path/to/output-dir\n"
        "Then run `body-models set garment-measurements /path/to/output-dir`."
    )


def _validate_shapes(path: Path, data: dict[str, np.ndarray], num_joints: int) -> None:
    num_vertices = data["mean_vertices"].shape[0]
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
    "PREPROCESSED_FILENAME",
    "GarmentMeasurementsAssets",
    "download_model",
    "load_model_data",
    "validate_path",
]
