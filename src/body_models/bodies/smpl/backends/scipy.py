"""SciPy sparse SMPL backend."""

import numpy as np
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.bodies.smpl.backends import core
from body_models.bodies.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: SmplWeights,
    shape: Float[np.ndarray, "*batch 10"],
    skip_vertices: bool = False,
) -> core.SmplIdentity:
    return core.prepare_identity(
        xp=np,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: SmplWeights,
    body_pose: Float[np.ndarray, "*batch 23 N"] | Float[np.ndarray, "*batch 23 3 3"],
    pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_joints: Float[np.ndarray, "*batch J 3"],
    skip_vertices: bool = False,
) -> core.SmplPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return core.prepare_pose(
        xp=np,
        posedirs=weights.posedirs,
        kinematic_fronts=weights.kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: SmplWeights,
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    skinning_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    lbs_weights = weights.lbs_weights
    if vertex_indices is not None:
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]
        lbs_weights = lbs_weights[vertex_indices]

    v_shaped = rest_vertices + pose_offsets
    v_posed = core.linear_blend_skinning(np, v_shaped, skinning_transforms, lbs_weights)
    return core.apply_global_transform(
        np,
        v_posed,
        rotation=global_rotation,
        translation=global_translation,
        rotation_type=rotation_type,
    )


def forward_skeleton(
    weights: SmplWeights,
    skeleton_transforms: Float[np.ndarray, "*batch J 4 4"],
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return core.forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        skeleton_transforms=skeleton_transforms,
        xp=np,
    )
