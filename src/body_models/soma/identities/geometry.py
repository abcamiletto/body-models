"""Geometry helpers for SOMA identity setup."""

import numpy as np
from jaxtyping import Float


def fit_rigid_transform(
    source: Float[np.ndarray, "V 3"],
    target: Float[np.ndarray, "V 3"],
) -> tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3"]]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    U, _S, Vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1] *= -1
        rotation = Vt.T @ U.T
    translation = target_mean - source_mean @ rotation.T
    return rotation.astype(np.float32), translation.astype(np.float32)
