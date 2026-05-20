"""NumPy SMPL-H backend."""

import numpy as np
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.smplh.backends.core import SmplhIdentity
from body_models.smplh.backends.core import forward_skeleton as _forward_skeleton
from body_models.smplh.backends.core import forward_vertices as _forward_vertices
from body_models.smplh.backends.core import prepare_identity as _prepare_identity
from body_models.smplh.io import SmplhWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def prepare_identity(
    weights: SmplhWeights,
    shape: Float[np.ndarray, "*batch 10"],
    skip_vertices: bool = False,
) -> SmplhIdentity:
    return _prepare_identity(
        xp=np,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: SmplhWeights,
    body_pose: Float[np.ndarray, "*batch 21 N"] | Float[np.ndarray, "*batch 21 3 3"],
    hand_pose: Float[np.ndarray, "*batch 30 N"] | Float[np.ndarray, "*batch 30 3 3"],
    pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
):
    return _forward_vertices(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        posedirs=weights.posedirs,
        lbs_weights=weights.lbs_weights,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        rest_vertices=rest_vertices,
        xp=np,
    )


def forward_skeleton(
    weights: SmplhWeights,
    body_pose: Float[np.ndarray, "*batch 21 N"] | Float[np.ndarray, "*batch 21 3 3"],
    hand_pose: Float[np.ndarray, "*batch 30 N"] | Float[np.ndarray, "*batch 30 3 3"],
    pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"] | None = None,
):
    return _forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        xp=np,
    )
