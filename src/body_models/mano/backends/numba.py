"""Numba MANO backend."""

import numpy as np
from jaxtyping import Float

from body_models import common
from body_models.mano.backends import core
from body_models.mano.io import ManoWeights
from body_models.rotations import RotationType
from body_models.smpl.backends import numba as smpl_numba

from body_models.mano.backends.numpy import forward_skeleton, prepare_identity

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def forward_vertices(
    weights: ManoWeights,
    hand_pose: Float[np.ndarray, "B 15 N"] | Float[np.ndarray, "B 15 3 3"],
    wrist_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
):
    posedirs = weights.posedirs
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        selected_vertices = np.asarray(vertex_indices)
        rest_vertices = rest_vertices[..., selected_vertices, :]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, selected_vertices].reshape(posedirs.shape[0], -1)
        joint_indices = joint_indices[selected_vertices]
        joint_weights = joint_weights[selected_vertices]

    pose_matrices, T_world = core._forward_core(
        xp=np,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        local_joint_offsets=local_joint_offsets,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
    )

    batch_shape = pose_matrices.shape[:-3]
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=np)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = rest_vertices + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    return smpl_numba.numba_skin(
        v_shaped,
        rest_joints,
        T_world,
        joint_indices,
        joint_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
