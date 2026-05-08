"""SciPy sparse SMPL-H backend."""

import numpy as np
from jaxtyping import Float

from body_models import common
from body_models.rotations import RotationType
from body_models.smpl.backends import scipy as smpl_scipy
from body_models.smplh.backends import core
from body_models.smplh.io import SmplhWeights

from body_models.smplh.backends.numpy import forward_skeleton

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplhWeights,
    shape: Float[np.ndarray, "B 10"],
    body_pose: Float[np.ndarray, "B 21 N"] | Float[np.ndarray, "B 21 3 3"],
    hand_pose: Float[np.ndarray, "B 30 N"] | Float[np.ndarray, "B 30 3 3"],
    pelvis_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    v_template = weights.v_template
    shapedirs = weights.shapedirs
    posedirs = weights.posedirs
    lbs_weights = weights.lbs_weights
    if vertex_indices is not None:
        selected_vertices = np.asarray(vertex_indices)
        v_template = v_template[selected_vertices]
        shapedirs = shapedirs[selected_vertices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, selected_vertices].reshape(posedirs.shape[0], -1)
        lbs_weights = lbs_weights[selected_vertices]

    v_t, j_t, pose_matrices, T_world = core._forward_core(
        xp=np,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        body_pose=body_pose,
        hand_pose=hand_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )

    batch_shape = pose_matrices.shape[:-3]
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=np)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    return smpl_scipy.sparse_skin(
        v_shaped,
        j_t,
        T_world,
        lbs_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
