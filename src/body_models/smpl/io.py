from pathlib import Path

import numpy as np

from .. import config


def get_model_path(model_path: Path | str | None, gender: str | None) -> Path:
    """Resolve SMPL model file path.

    Args:
        model_path: Direct path to model file, or None to use config.
        gender: Gender for config lookup when model_path is None.

    Returns:
        Path to the model file.

    Raises:
        ValueError: If model_path is a directory (no longer supported) or
            if neither model_path nor gender is provided.
        FileNotFoundError: If the model file cannot be found.
    """
    if model_path is not None:
        model_path = Path(model_path)

        if model_path.is_dir():
            raise ValueError(
                f"Directory paths are no longer supported: {model_path}\n"
                "Please provide a direct path to the model file, e.g.:\n"
                f"  SMPL(model_path='{model_path}/SMPL_NEUTRAL.npz')"
            )

        if model_path.is_file():
            return model_path

        raise FileNotFoundError(f"SMPL model file not found: {model_path}")

    # model_path is None, lookup from config using gender
    if gender is None:
        raise ValueError(
            "Either model_path or gender must be provided.\n"
            "Examples:\n"
            "  SMPL(model_path='/path/to/SMPL_NEUTRAL.npz')\n"
            "  SMPL(gender='neutral')  # Uses smpl-neutral config key"
        )

    config_key = f"smpl-{gender}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            f"SMPL model not found. Download from https://smpl.is.tue.mpg.de/ "
            f"and run: body-models set smpl-{gender} /path/to/SMPL_{gender.upper()}.npz"
        )

    return resolved_path


def load_model_data(model_path: Path) -> dict:
    """Load SMPL model data from a .pkl or .npz file."""
    if model_path.suffix == ".npz":
        return dict(np.load(model_path, allow_pickle=True))

    import pickle

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as e:
        if "chumpy" in str(e):
            npz_path = model_path.with_suffix(".npz")
            raise RuntimeError(
                f"This SMPL pkl file requires chumpy to load. "
                f"Convert it to npz format first:\n\n"
                f"  uvx --from body-models convert-smpl-pkl {model_path} {npz_path}\n\n"
                f"Then use the npz file instead."
            ) from None
        raise

    if hasattr(model_data["J_regressor"], "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


def compute_kinematic_fronts(parents: np.ndarray) -> list[tuple[list[int], list[int]]]:
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


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplify mesh using quadric decimation."""
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map
