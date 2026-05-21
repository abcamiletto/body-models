"""SciPy sparse MANO backend."""

import numpy as np
from jaxtyping import Float

from body_models.mano.io import ManoWeights
from body_models.rotations import RotationType
from body_models.smpl.backends import scipy as smpl_scipy

from body_models.mano.backends.numpy import forward_skeleton, prepare_identity, prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: ManoWeights,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    joint_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
):
    lbs_weights = weights.lbs_weights
    v_shaped = rest_vertices + pose_offsets
    if vertex_indices is not None:
        selected_vertices = np.asarray(vertex_indices)
        v_shaped = v_shaped[..., selected_vertices, :]
        lbs_weights = lbs_weights[selected_vertices]

    return smpl_scipy.sparse_skin(
        v_shaped,
        rest_joints,
        joint_transforms,
        lbs_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
