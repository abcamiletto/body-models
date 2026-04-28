from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Int

from .. import config
from ..common import simplify_mesh

PathLike = Path | str

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

SMPL_JOINT_NAMES = [
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
    "left_hand",
    "right_hand",
]


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
            f"SMPL model not found. Download from https://smpl.is.tue.mpg.de/ "
            f"and run: body-models set smpl-{gender} /path/to/SMPL_{gender.upper()}.pkl"
        )

    return validate_path(resolved_path)


def load_model_data(model_path: Path) -> dict:
    """Load SMPL model data from a .pkl or .npz file."""
    model_data = (
        dict(np.load(model_path, allow_pickle=True)) if model_path.suffix == ".npz" else _load_smpl_pkl(model_path)
    )

    if hasattr(model_data["J_regressor"], "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


def _load_smpl_pkl(model_path: Path) -> dict:
    """Load legacy SMPL pickles without requiring chumpy at runtime."""
    import pickle

    class _ChumpyPlaceholder:
        def __setstate__(self, state):
            if isinstance(state, dict) and "x" in state:
                self.data = np.asarray(state["x"])
                return
            if isinstance(state, dict):
                for value in state.values():
                    if isinstance(value, np.ndarray):
                        self.data = value
                        return
            self.data = state

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "scipy.sparse.csc" and name == "csc_matrix":
                from scipy.sparse import csc_matrix

                return csc_matrix
            if module.startswith("chumpy"):
                return _ChumpyPlaceholder
            return super().find_class(module, name)

    with open(model_path, "rb") as f:
        data = _CompatUnpickler(f, encoding="latin1").load()

    return {
        key: value.data
        if isinstance(value, _ChumpyPlaceholder)
        else value.toarray()
        if hasattr(value, "toarray")
        else value
        for key, value in data.items()
    }


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


__all__ = ["SMPL_JOINT_NAMES", "get_model_path", "load_model_data", "compute_kinematic_fronts", "simplify_mesh"]
