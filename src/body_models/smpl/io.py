from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from body_models import config
from body_models.cache import get_cached_path
from body_models.common import load_model_dict, simplify_mesh, validate_simplify
from body_models.smpl.download import SMPL_FILES

PathLike = Path | str
Array = Any

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class SmplWeights:
    v_template: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 24"]
    lbs_joint_indices: Int[Array, "V K"]
    lbs_joint_weights: Float[Array, "V K"]
    shapedirs: Float[Array, "V 3 S"]
    posedirs: Float[Array, "P V*3"]
    j_template: Float[Array, "24 3"]
    j_shapedirs: Float[Array, "24 3 S"]
    parents: list[int]
    kinematic_fronts: list[Front]


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
        resolved_path = get_cached_path(SMPL_FILES[config_key])

    if resolved_path is None:
        raise FileNotFoundError(
            f"SMPL model not found. Download from https://smpl.is.tue.mpg.de/ "
            f"and run: body-models set smpl-{gender} /path/to/SMPL_{gender.upper()}.pkl"
        )

    return validate_path(resolved_path)


def load_model_data(model_path: Path, simplify: float = 1.0) -> SmplWeights:
    """Load SMPL model data from a .pkl or .npz file."""
    validate_simplify(simplify)
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

    lbs_joint_indices, lbs_joint_weights = compute_sparse_lbs_weights(lbs_weights)

    return SmplWeights(
        v_template=v_template,
        faces=faces,
        lbs_weights=lbs_weights,
        lbs_joint_indices=lbs_joint_indices,
        lbs_joint_weights=lbs_joint_weights,
        shapedirs=shapedirs,
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parent_list,
        kinematic_fronts=compute_kinematic_fronts(parents),
    )


def compute_sparse_lbs_weights(
    lbs_weights: Float[np.ndarray, "V J"],
) -> tuple[Int[np.ndarray, "V K"], Float[np.ndarray, "V K"]]:
    counts = (np.abs(lbs_weights) > 1e-8).sum(axis=1)
    indices = np.full((lbs_weights.shape[0], int(counts.max())), -1, dtype=np.int64)
    weights = np.zeros(indices.shape, dtype=lbs_weights.dtype)

    for vertex, row in enumerate(lbs_weights):
        active = np.flatnonzero(np.abs(row) > 1e-8)
        indices[vertex, : len(active)] = active
        weights[vertex, : len(active)] = row[active]

    return indices, weights


def compute_kinematic_fronts(parents: Int[np.ndarray, "J"]) -> list[Front]:
    """Compute kinematic fronts for batched FK."""
    n_joints = len(parents)
    depths = [-1] * n_joints
    depths[0] = 0

    for i in range(1, n_joints):
        d = 0
        j = i
        while j != 0:
            j = int(parents[j])
            d += 1
        depths[i] = d

    max_depth = max(depths)
    fronts: list[Front] = []
    for d in range(0, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        if d == 0:
            parent_indices = [-1] * len(joints)
        else:
            parent_indices = [int(parents[j]) for j in joints]
        fronts.append((joints, parent_indices))

    return fronts
