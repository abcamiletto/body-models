"""I/O utilities for MANO model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Int

from .. import config
from ..common import simplify_mesh
from ..smpl.io import _load_smpl_pkl

PathLike = Path | str

__all__ = [
    "validate_path",
    "get_model_path",
    "get_joint_names",
    "load_model_data",
    "compute_kinematic_fronts",
    "simplify_mesh",
]
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

MANO_JOINT_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected a MANO model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model file not found: {model_path}")
    if model_path.suffix not in {".npz", ".pkl"}:
        raise ValueError(f"Expected a MANO .npz or .pkl file, got: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None, side: Literal["right", "left"] | None) -> Path:
    if model_path is not None:
        if side is not None:
            raise ValueError("side is only supported when model_path is not provided.")
        return validate_path(model_path)

    if side is None:
        raise ValueError("Either model_path or side must be provided.")

    config_key = f"mano-{side}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            "MANO model not found. Download from https://mano.is.tue.mpg.de/ "
            f"and run: body-models set mano-{side} /path/to/MANO_{side.upper()}.pkl"
        )

    return validate_path(resolved_path)


def load_model_data(path: Path) -> dict:
    """Load MANO model data from .pkl or .npz file."""
    data = dict(np.load(path, allow_pickle=True)) if path.suffix == ".npz" else _load_smpl_pkl(path)
    if hasattr(data["J_regressor"], "toarray"):
        data["J_regressor"] = data["J_regressor"].toarray()
    v_template = np.asarray(data["v_template"])
    if np.asarray(data["shapedirs"]).ndim == 1:
        data["shapedirs"] = np.asarray(data["shapedirs"]).reshape(v_template.shape[0], 3, -1)
    if np.asarray(data["posedirs"]).ndim == 1:
        data["posedirs"] = np.asarray(data["posedirs"]).reshape(v_template.shape[0], 3, -1)
    return data


def get_joint_names(model_data: dict) -> list[str]:
    """Extract ordered MANO joint names from model data."""
    if "joint2num" not in model_data:
        return list(MANO_JOINT_NAMES)
    joint2num = model_data["joint2num"].item()
    return [name for name, _ in sorted(joint2num.items(), key=lambda item: int(item[1]))]


def compute_kinematic_fronts(parents: Int[np.ndarray, "J"]) -> list[Front]:
    """Compute kinematic fronts for batched FK.

    Returns list of (joint_indices, parent_indices) tuples, one per depth level.
    Joints at the same depth can be processed in parallel.
    """
    n_joints = len(parents)
    depths = [-1] * n_joints
    depths[0] = 0  # Root

    # Compute depth of each joint
    for i in range(1, n_joints):
        d = 0
        j = i
        while j != 0:
            j = int(parents[j])
            d += 1
        depths[i] = d

    # Group joints by depth
    max_depth = max(depths)
    fronts: list[Front] = []

    # Add root level
    root_joints = [i for i in range(n_joints) if depths[i] == 0]
    fronts.append((root_joints, [-1] * len(root_joints)))

    for d in range(1, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        parent_indices = [int(parents[j]) for j in joints]
        fronts.append((joints, parent_indices))

    return fronts
