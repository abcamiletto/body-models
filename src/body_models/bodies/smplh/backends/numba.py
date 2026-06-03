"""Numba SMPL-H backend."""

import numpy as np
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.bodies.smpl.backends import core as smpl_core
from body_models.bodies.smplh.io import SmplhWeights

from body_models.bodies.smplh.backends.numpy import forward_skeleton, prepare_identity, prepare_pose

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def forward_vertices(
    weights: SmplhWeights,
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    skinning_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    v_shaped = rest_vertices + pose_offsets
    lbs_weights = weights.lbs_weights
    if vertex_indices is not None:
        vertex_index_array = np.asarray(vertex_indices)
        v_shaped = v_shaped[..., vertex_index_array, :]
        lbs_weights = lbs_weights[vertex_index_array]

    v_posed = smpl_core.linear_blend_skinning(np, v_shaped, skinning_transforms, lbs_weights)
    return smpl_core.apply_global_transform(
        np,
        v_posed,
        rotation=global_rotation,
        translation=global_translation,
        rotation_type=rotation_type,
    )
