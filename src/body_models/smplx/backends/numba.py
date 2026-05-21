"""Numba SMPL-X backend."""

import numpy as np
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.smpl.backends import numba as smpl_numba
from body_models.smplx.io import SmplxWeights

from body_models.smplx.backends.numpy import forward_skeleton, prepare_identity, prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: SmplxWeights,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    joint_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
):
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    v_shaped = rest_vertices + pose_offsets
    if vertex_indices is not None:
        selected_vertices = np.asarray(vertex_indices)
        v_shaped = v_shaped[..., selected_vertices, :]
        joint_indices = joint_indices[selected_vertices]
        joint_weights = joint_weights[selected_vertices]
    return smpl_numba.numba_skin(
        v_shaped,
        rest_joints,
        joint_transforms,
        joint_indices,
        joint_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
