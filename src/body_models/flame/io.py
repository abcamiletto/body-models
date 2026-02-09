from pathlib import Path

import numpy as np
from jaxtyping import Int

from .. import config
from ..common import simplify_mesh

__all__ = ["get_model_path", "load_model_data", "compute_kinematic_fronts", "simplify_mesh"]


def get_model_path(model_path: Path | str | None) -> Path:
    if model_path is None:
        model_path = config.get_model_path("flame")

    if model_path is None:
        raise FileNotFoundError(
            "FLAME model not found. Download from https://flame.is.tue.mpg.de/ "
            "and run: body-models set flame /path/to/flame"
        )

    model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        # Try common FLAME model filenames
        for candidate_name in ["FLAME_NEUTRAL.pkl", "FLAME_NEUTRAL.npz", "flame2023.pkl", "generic_model.pkl"]:
            candidate = model_path / candidate_name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"FLAME model not found in {model_path}")


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


def compute_kinematic_fronts(parents: Int[np.ndarray, "J"]) -> list[tuple[list[int], list[int]]]:
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
    fronts = []
    for d in range(0, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        if d == 0:
            parent_indices = [-1] * len(joints)
        else:
            parent_indices = [int(parents[j]) for j in joints]
        fronts.append((joints, parent_indices))

    return fronts


