"""I/O utilities for SMPL-H model."""

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

SMPLH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected an SMPLH model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPLH model file not found: {model_path}")
    if model_path.suffix not in {".npz", ".pkl"}:
        raise ValueError(f"Expected an SMPLH .npz or .pkl file, got: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None, gender: Literal["neutral", "male", "female"] | None) -> Path:
    if model_path is not None:
        if gender is not None:
            raise ValueError("gender is only supported when model_path is not provided.")
        return validate_path(model_path)

    if gender is None:
        raise ValueError("Either model_path or gender must be provided.")

    config_key = f"smplh-{gender}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            "SMPLH model not found. Download from https://mano.is.tue.mpg.de/ "
            f"and run: body-models set smplh-{gender} /path/to/smplh/{gender}/model.npz"
        )

    return validate_path(resolved_path)


def load_model_data(path: Path) -> dict:
    """Load SMPL-H model data from .pkl or .npz file."""
    data = dict(np.load(path, allow_pickle=True)) if path.suffix == ".npz" else _load_smpl_pkl(path)
    if hasattr(data["J_regressor"], "toarray"):
        data["J_regressor"] = data["J_regressor"].toarray()
    return data


def get_joint_names(model_data: dict) -> list[str]:
    """Extract ordered SMPL-H joint names from model data."""
    if "joint2num" not in model_data:
        return list(SMPLH_JOINT_NAMES)
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
