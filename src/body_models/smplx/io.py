"""I/O utilities for SMPL-X model."""

from pathlib import Path

import numpy as np
from jaxtyping import Int

from .. import config
from ..common import simplify_mesh

__all__ = ["get_model_path", "load_model_data", "compute_kinematic_fronts", "simplify_mesh"]


def get_model_path(model_path: Path | str | None, gender: str | None) -> Path:
    """Resolve SMPL-X model file path.

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
                f"  SMPLX(model_path='{model_path}/SMPLX_NEUTRAL.npz')"
            )

        if model_path.is_file():
            return model_path

        raise FileNotFoundError(f"SMPLX model file not found: {model_path}")

    # model_path is None, lookup from config using gender
    if gender is None:
        raise ValueError(
            "Either model_path or gender must be provided.\n"
            "Examples:\n"
            "  SMPLX(model_path='/path/to/SMPLX_NEUTRAL.npz')\n"
            "  SMPLX(gender='neutral')  # Uses smplx-neutral config key"
        )

    config_key = f"smplx-{gender}"
    resolved_path = config.get_model_path(config_key)

    if resolved_path is None:
        raise FileNotFoundError(
            f"SMPLX model not found. Download from https://smpl-x.is.tue.mpg.de/ "
            f"and run: body-models set smplx-{gender} /path/to/SMPLX_{gender.upper()}.npz"
        )

    return resolved_path


def load_model_data(path: Path) -> dict:
    """Load SMPL-X model data from .npz file."""
    return dict(np.load(path, allow_pickle=True))


def compute_kinematic_fronts(parents: Int[np.ndarray, "J"]) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK.

    Returns list of (joint_indices, parent_indices) tuples, one per depth level.
    Joints at the same depth can be processed in parallel.
    """
    n_joints = len(parents)
    depths = [-1] * n_joints
    depths[0] = 0  # Root

    # Compute depth of each joint
    for i in range(1, n_joints):
        d = 0
        j = i
        while j != 0:
            j = int(parents[j])
            d += 1
        depths[i] = d

    # Group joints by depth
    max_depth = max(depths)
    fronts: list[tuple[list[int], list[int]]] = []

    # Add root level
    root_joints = [i for i in range(n_joints) if depths[i] == 0]
    fronts.append((root_joints, [-1] * len(root_joints)))

    for d in range(1, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        parent_indices = [int(parents[j]) for j in joints]
        fronts.append((joints, parent_indices))

    return fronts
