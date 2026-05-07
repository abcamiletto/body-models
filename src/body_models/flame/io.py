from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models import config
from body_models.common import simplify_mesh
from body_models.common.chumpy_fix import load_model_dict

PathLike = Path | str
Array = Any

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

__all__ = ["load_model_data"]


@dataclass(frozen=True)
class FlameWeights:
    v_template: Float[Array, "V 3"]
    v_template_full: Float[Array, "V_full 3"]
    faces: Int[Array, "F 3"]
    lbs_weights: Float[Array, "V 5"]
    shapedirs: Float[Array, "V 3 S"]
    shapedirs_full: Float[Array, "V_full 3 S"]
    exprdirs: Float[Array, "V 3 E"]
    exprdirs_full: Float[Array, "V_full 3 E"]
    posedirs: Float[Array, "P V*3"]
    J_regressor: Float[Array, "5 V_full"]
    parents: list[int]
    kinematic_fronts: list[Front]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_dir():
        raise ValueError(f"Expected a FLAME model file, got directory: {model_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"FLAME model file not found: {model_path}")
    if model_path.suffix not in {".pkl", ".npz"}:
        raise ValueError(f"Expected a FLAME .pkl or .npz file, got: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("flame")

    if model_path is None:
        raise FileNotFoundError(
            "FLAME model not found. Download from https://flame.is.tue.mpg.de/ "
            "and run: body-models set flame /path/to/FLAME_NEUTRAL.pkl"
        )

    return validate_path(model_path)


def load_model_data(model_path: Path, simplify: float = 1.0) -> FlameWeights:
    """Load FLAME model data from a .pkl or .npz file."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    model_data = load_model_dict(model_path)

    v_template_full = np.asarray(model_data["v_template"], dtype=np.float32)
    faces = np.asarray(model_data["f"], dtype=np.int32)
    lbs_weights = np.asarray(model_data["weights"], dtype=np.float32)
    shapedirs_full = np.asarray(model_data["shapedirs"], dtype=np.float32)
    posedirs = np.asarray(model_data["posedirs"], dtype=np.float32)
    J_regressor = np.asarray(model_data["J_regressor"], dtype=np.float32)
    parents = np.asarray(model_data["kintree_table"][0], dtype=np.int64)
    parents[0] = -1

    v_template = v_template_full
    shapedirs = shapedirs_full
    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
        lbs_weights = lbs_weights[vertex_map]
        shapedirs = shapedirs_full[vertex_map]
        posedirs = posedirs[vertex_map]

    return FlameWeights(
        v_template=v_template,
        v_template_full=v_template_full,
        faces=faces,
        lbs_weights=lbs_weights,
        shapedirs=shapedirs[:, :, :300],
        shapedirs_full=shapedirs_full[:, :, :300],
        exprdirs=shapedirs[:, :, 300:],
        exprdirs_full=shapedirs_full[:, :, 300:],
        posedirs=posedirs.reshape(-1, posedirs.shape[-1]).T,
        J_regressor=J_regressor,
        parents=parents.tolist(),
        kinematic_fronts=compute_kinematic_fronts(parents),
    )


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
