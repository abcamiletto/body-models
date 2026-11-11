"""I/O utilities for MANO model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._common import chumpy_fix, kinematics, simplify_mesh
from body_models._common.skinning import CompactSkinning
from body_models.mano._constants import MANO_JOINT_NAMES

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class ManoAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 16"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "16 3"]
    j_shapedirs: Float[Array, "16 3 S"]
    hand_mean: Float[Array, "45"]
    kinematic_tree: kinematics.KinematicTree
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    return config.validate_file(model_path, "MANO", (".npz", ".pkl"))


def load_model_data(
    path: Path,
    flat_hand_mean: bool = False,
    flip_shapedirs: bool = False,
    simplify: float = 1.0,
) -> ManoAssets:
    """Load MANO model data from .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    data = chumpy_fix.load_model_dict(path)
    v_template = np.asarray(data["v_template"])
    if np.asarray(data["shapedirs"]).ndim == 1:
        data["shapedirs"] = np.asarray(data["shapedirs"]).reshape(v_template.shape[0], 3, -1)
    if np.asarray(data["posedirs"]).ndim == 1:
        data["posedirs"] = np.asarray(data["posedirs"]).reshape(v_template.shape[0], 3, -1)

    model_template = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)
    lbs_weights = np.asarray(data["weights"], dtype=np.float32)
    model_dirs = np.asarray(data["shapedirs"], dtype=np.float32)
    if flip_shapedirs:
        # The official left model ships the right model's shapedirs although its template
        # is mirrored in x, so flip their x component to mirror the shape space too.
        model_dirs = model_dirs * np.array([[-1.0], [1.0], [1.0]], dtype=np.float32)
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

    hand_mean = np.asarray(data["hands_mean"], dtype=np.float32)
    if flat_hand_mean:
        hand_mean = np.zeros_like(hand_mean)

    lbs_joint_indices, lbs_joint_weights = kinematics.compute_sparse_skin_weights(lbs_weights)

    return ManoAssets(
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
    """Extract ordered MANO joint names from model data."""
    if "joint2num" not in model_data:
        return list(MANO_JOINT_NAMES)
    joint2num = model_data["joint2num"]
    if isinstance(joint2num, np.ndarray):
        joint2num = joint2num.item()
    return [name for name, _ in sorted(joint2num.items(), key=lambda item: int(item[1]))]
