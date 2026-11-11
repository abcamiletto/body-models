"""I/O utilities for SMPL-H model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._common import chumpy_fix, kinematics, simplify_mesh
from body_models._common.skinning import CompactSkinning
from body_models.smplh._constants import SMPLH_JOINT_NAMES

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplhAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 52"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "52 3"]
    j_shapedirs: Float[Array, "52 3 S"]
    hand_mean: Float[Array, "2 45"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    return config.validate_file(model_path, "SMPL-H", (".npz", ".pkl"))


def load_model_data(path: Path, flat_hand_mean: bool = False, simplify: float = 1.0) -> SmplhAssets:
    """Load SMPL-H model data from .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    data = chumpy_fix.load_model_dict(path)

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
        v_template, faces, vertex_map = simplify_mesh.simplify_mesh(model_template, faces, target_faces)
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

    lbs_joint_indices, lbs_joint_weights = kinematics.compute_sparse_skin_weights(lbs_weights)

    return SmplhAssets(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(lbs_joint_indices, lbs_joint_weights),
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=joint_regressor @ model_template,
        j_shapedirs=np.einsum("jv,vds->jds", joint_regressor, model_dirs),
        hand_mean=hand_mean,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
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
