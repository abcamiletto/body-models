"""NumPy MHR backend."""

import numpy as np
from jaxtyping import Float

from body_models.mhr.backends.core import forward_skeleton as _forward_skeleton
from body_models.mhr.backends.core import forward_vertices as _forward_vertices
from body_models.mhr.io import MhrWeights

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: MhrWeights,
    shape: Float[np.ndarray, "B 45"],
    pose: Float[np.ndarray, "B 204"],
    expression: Float[np.ndarray, "B 72"] | None = None,
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
):
    return _forward_vertices(
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        skin_weights=weights.skin_weights,
        skin_indices=weights.skin_indices,
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        bind_inv_linear=weights.bind_inv_linear,
        bind_inv_translation=weights.bind_inv_translation,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        expr_dim=72,
        shape=shape,
        pose=pose,
        expression=expression,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        corrective_W1=weights.corrective_W1,
        corrective_W2=weights.corrective_W2,
        xp=np,
    )


def forward_skeleton(
    weights: MhrWeights,
    shape: Float[np.ndarray, "B 45"],
    pose: Float[np.ndarray, "B 204"],
    expression: Float[np.ndarray, "B 72"] | None = None,
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
):
    return _forward_skeleton(
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        xp=np,
    )
