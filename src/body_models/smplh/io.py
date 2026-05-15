"""I/O utilities for SMPL-H model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from body_models import config
from body_models.common import simplify_mesh
from body_models.common.chumpy_fix import load_model_dict
from body_models.smpl.io import compute_sparse_lbs_weights
from body_models.smplh.constants import SMPLH_JOINT_NAMES

PathLike = Path | str
Array = Any
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplhWeights:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 52"]
    lbs_joint_indices: Int[Array, "V K"]
    lbs_joint_weights: Float[Array, "V K"]
    shapedirs: Float[Array, "V 3 S"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "52 3"]
    j_shapedirs: Float[Array, "52 3 S"]
    hand_mean: Float[Array, "2 45"]
    parents: list[int]
    kinematic_fronts: list[Front]
    joint_names: list[str]


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


def load_model_data(path: Path, flat_hand_mean: bool = False, simplify: float = 1.0) -> SmplhWeights:
    """Load SMPL-H model data from .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    data = load_model_dict(path)

    model_template = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    lbs_weights = np.asarray(data["weights"], dtype=np.float32)
    model_dirs = np.asarray(data["shapedirs"], dtype=np.float32)
    posedirs = np.asarray(data["posedirs"], dtype=np.float32)
    joint_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
    parents = np.asarray(data["kintree_table"][0], dtype=np.int64)
    parents[0] = -1

    v_template = model_template
    shapedirs = model_dirs
    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh(model_template, faces, target_faces)
        lbs_weights = lbs_weights[vertex_map]
        shapedirs = model_dirs[vertex_map]
        posedirs = posedirs[vertex_map]

    if flat_hand_mean:
        hand_mean = np.zeros((2, 45), dtype=np.float32)
    elif "hands_meanl" in data and "hands_meanr" in data:
        hand_mean = np.stack(
            [
                np.asarray(data["hands_meanl"], dtype=np.float32),
                np.asarray(data["hands_meanr"], dtype=np.float32),
            ]
        )
    else:
        hand_mean = np.zeros((2, 15, 3), dtype=np.float32)
        hand_mean[:, :, 0] = 0.55
        hand_mean = hand_mean.reshape(2, 45)

    lbs_joint_indices, lbs_joint_weights = compute_sparse_lbs_weights(lbs_weights)

    return SmplhWeights(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        lbs_joint_indices=lbs_joint_indices,
        lbs_joint_weights=lbs_joint_weights,
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=joint_regressor @ model_template,
        j_shapedirs=np.einsum("jv,vds->jds", joint_regressor, model_dirs),
        hand_mean=hand_mean,
        parents=parents.tolist(),
        kinematic_fronts=compute_kinematic_fronts(parents),
        joint_names=get_joint_names(data),
    )


def get_joint_names(model_data: dict) -> list[str]:
    """Extract ordered SMPL-H joint names from model data."""
    if "joint2num" not in model_data:
        return list(SMPLH_JOINT_NAMES)
    joint2num = model_data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
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
