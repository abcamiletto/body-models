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
class FlameAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 5"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    exprdirs: Float[Array, "V 3 E"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "5 3"]
    j_shapedirs: Float[Array, "5 3 S"]
    j_exprdirs: Float[Array, "5 3 E"]
    kinematic_tree: kinematics.KinematicTree


def validate_path(model_path: PathLike) -> Path:
    return config.validate_file(model_path, "FLAME", (".pkl", ".npz"))


def load_model_data(model_path: Path, simplify: float = 1.0) -> FlameAssets:
    """Load FLAME model data from a .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    model_data = chumpy_fix.load_model_dict(model_path)

    model_template = np.asarray(model_data["v_template"], dtype=np.float32)
    faces = np.asarray(model_data["f"], dtype=np.int32)
    lbs_weights = np.asarray(model_data["weights"], dtype=np.float32)
    model_dirs = np.asarray(model_data["shapedirs"], dtype=np.float32)
    posedirs = np.asarray(model_data["posedirs"], dtype=np.float32)
    joint_regressor = np.asarray(model_data["J_regressor"], dtype=np.float32)
    parents = np.asarray(model_data["kintree_table"][0], dtype=np.int64)
    parents[0] = -1

    j_template = joint_regressor @ model_template
    j_shapedirs = np.einsum("jv,vds->jds", joint_regressor, model_dirs[:, :, :300])
    j_exprdirs = np.einsum("jv,vde->jde", joint_regressor, model_dirs[:, :, 300:])

    v_template = model_template
    shapedirs = model_dirs
    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh.simplify_mesh(model_template, faces, target_faces)
        lbs_weights = lbs_weights[vertex_map]
        shapedirs = model_dirs[vertex_map]
        posedirs = posedirs[vertex_map]

    lbs_joint_indices, lbs_joint_weights = kinematics.compute_sparse_skin_weights(lbs_weights)

    return FlameAssets(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(lbs_joint_indices, lbs_joint_weights),
        shapedirs=shapedirs[:, :, :300],
        exprdirs=shapedirs[:, :, 300:],
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        j_exprdirs=j_exprdirs,
        kinematic_tree=kinematics.KinematicTree.from_parents(parents),
    )
