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

    model_data = _load_flame_pkl(model_path)

    # Handle scipy sparse matrices
    if hasattr(model_data.get("J_regressor"), "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


def _load_flame_pkl(model_path: Path) -> dict:
    """Load FLAME pickles without requiring chumpy at runtime."""
    import pickle

    class _ChumpyPlaceholder:
        def __setstate__(self, state):
            if isinstance(state, dict) and "x" in state:
                self.data = np.asarray(state["x"])
                return
            if isinstance(state, dict) and "a" in state and "idxs" in state:
                source = state["a"]
                source_data = source.data if isinstance(source, _ChumpyPlaceholder) else np.asarray(source)
                data = np.asarray(source_data).reshape(-1)[np.asarray(state["idxs"], dtype=np.int64)]
                if "preferred_shape" in state:
                    data = data.reshape(tuple(state["preferred_shape"]))
                self.data = data
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
