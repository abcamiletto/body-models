"""I/O utilities for SMPL-X model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._common import chumpy_fix, kinematics, simplify_mesh
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplxAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 55"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    exprdirs: Float[Array, "V 3 E"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "55 3"]
    j_shapedirs: Float[Array, "55 3 S"]
    j_exprdirs: Float[Array, "55 3 E"]
    hand_mean: Float[Array, "2 45"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    return config.validate_file(model_path, "SMPL-X", (".npz", ".pkl"))


def load_model_data(path: Path, flat_hand_mean: bool = False, simplify: float = 1.0) -> SmplxAssets:
    """Load SMPL-X model data from .pkl or .npz file."""
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

    hand_mean = np.stack(
        [
            np.asarray(data["hands_meanl"], dtype=np.float32),
            np.asarray(data["hands_meanr"], dtype=np.float32),
        ]
    )
    if flat_hand_mean:
        hand_mean = np.zeros_like(hand_mean)

    lbs_joint_indices, lbs_joint_weights = kinematics.compute_sparse_skin_weights(lbs_weights)

    return SmplxAssets(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(lbs_joint_indices, lbs_joint_weights),
        shapedirs=shapedirs[:, :, :300],
        exprdirs=shapedirs[:, :, 300:400],
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=joint_regressor @ model_template,
        j_shapedirs=np.einsum("jv,vds->jds", joint_regressor, model_dirs[:, :, :300]),
        j_exprdirs=np.einsum("jv,vde->jde", joint_regressor, model_dirs[:, :, 300:400]),
        hand_mean=hand_mean,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
        joint_names=get_joint_names(data),
    )


def get_joint_names(model_data: dict) -> list[str]:
    """Extract ordered SMPL-X joint names from model data."""
    joint2num = model_data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
    return [name for name, _ in sorted(joint2num.items(), key=lambda item: int(item[1]))]
