from pathlib import Path

import numpy as np
from jaxtyping import Int

from .. import config
from ..common import simplify_mesh

PathLike = Path | str

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

FLAME_JOINT_NAMES = ["root", "neck", "jaw", "left_eye", "right_eye"]

__all__ = ["FLAME_JOINT_NAMES", "get_model_path", "load_model_data", "compute_kinematic_fronts", "simplify_mesh"]


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


def load_model_data(model_path: Path) -> dict:
    """Load FLAME model data from a .pkl or .npz file."""
    if model_path.suffix == ".npz":
        return dict(np.load(model_path, allow_pickle=True))

    import pickle

    with open(model_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    # Handle scipy sparse matrices
    if hasattr(model_data.get("J_regressor"), "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


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
