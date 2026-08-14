from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from body_models import _config as config
from body_models._common import compute_sparse_skin_weights, kinematics, simplify_mesh
from body_models._common.chumpy_fix import load_model_dict
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
Array = Any

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplAssets:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 24"]
    compact_skinning: CompactSkinning
    shapedirs: Float[Array, "V 3 S"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "24 3"]
    j_shapedirs: Float[Array, "24 3 S"]
    kinematic_tree: kinematics.KinematicTree


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected an SMPL model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL model file not found: {model_path}")
    if model_path.suffix not in {".pkl", ".npz"}:
        raise ValueError(f"Expected an SMPL .pkl or .npz file, got: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None, gender: Literal["neutral", "male", "female"] | None) -> Path:
    if model_path is not None:
        if gender is not None:
            raise ValueError("gender is only supported when model_path is not provided.")
        return validate_path(model_path)

    if gender is None:
        raise ValueError("Either model_path or gender must be provided.")

    config_key = f"smpl-{gender}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            f"SMPL model not found. Run `body-models download smpl` or configure an existing file with "
            f"`body-models set smpl-{gender} /path/to/SMPL_{gender.upper()}.pkl`."
        )

    return validate_path(resolved_path)


def load_model_data(model_path: Path, simplify: float = 1.0) -> SmplAssets:
    """Load SMPL model data from a .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    model_data = load_model_dict(model_path)

    parents = np.asarray(model_data["kintree_table"][0], dtype=np.int64)
    parents[0] = -1
    parent_list = parents.tolist()

    v_template = np.asarray(model_data["v_template"], dtype=np.float32)
    faces = np.asarray(model_data["f"], dtype=np.int32)
    lbs_weights = np.asarray(model_data["weights"], dtype=np.float32)
    shapedirs = np.asarray(model_data["shapedirs"], dtype=np.float32)
    posedirs = np.asarray(model_data["posedirs"], dtype=np.float32)
    J_regressor = np.asarray(model_data["J_regressor"], dtype=np.float32)
    j_template = J_regressor @ v_template
    j_shapedirs = np.einsum("jv,vds->jds", J_regressor, shapedirs)

    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh(v_template, faces, target_faces)
        lbs_weights = lbs_weights[vertex_map]
        shapedirs = shapedirs[vertex_map]
        posedirs = posedirs[vertex_map]

    lbs_joint_indices, lbs_joint_weights = compute_sparse_skin_weights(lbs_weights)

    return SmplAssets(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        compact_skinning=CompactSkinning(lbs_joint_indices, lbs_joint_weights),
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        kinematic_tree=kinematics.KinematicTree.from_parents(parent_list),
    )
