"""NumPy SKEL backend."""

import numpy as np
from jaxtyping import Float

from body_models.skel.backends.core import SkelIdentity, SkelPreparedPose
from body_models.skel.backends.core import forward_skeleton as _forward_skeleton
from body_models.skel.backends.core import forward_vertices as _forward_vertices
from body_models.skel.backends.core import prepare_identity as _prepare_identity
from body_models.skel.backends.core import prepare_pose as _prepare_pose
from body_models.skel.io import SkelWeights

__all__ = ["forward_skeleton", "forward_vertices", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: SkelWeights,
    shape: Float[np.ndarray, "*batch 10"],
    skip_vertices: bool = False,
) -> SkelIdentity:
    return _prepare_identity(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parent=weights.parent,
        shape=shape,
        skip_vertices=skip_vertices,
        xp=np,
    )


def prepare_pose(
    weights: SkelWeights,
    pose: Float[np.ndarray, "*batch 46"],
    *,
    rest_joints: Float[np.ndarray, "*batch 24 3"],
    local_joint_offsets: Float[np.ndarray, "*batch 24 3"],
    skip_vertices: bool = False,
) -> SkelPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        all_axes=weights.all_axes,
        rotation_indices=weights.rotation_indices,
        apose_R=weights.apose_R,
        apose_t=weights.apose_t,
        per_joint_rot=weights.per_joint_rot,
        child=weights.child,
        fixed_orientation_joints=weights.fixed_orientation_joints,
        scapula_r_axes=weights.scapula_r_axes,
        scapula_l_axes=weights.scapula_l_axes,
        spine_axes=weights.spine_axes,
        parents=weights.parents,
        num_joints_smpl=weights.num_joints_smpl,
        posedirs=weights.posedirs,
        pose=pose,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        skip_vertices=skip_vertices,
        xp=np,
    )


def forward_vertices(
    weights: SkelWeights,
    global_rotation: Float[np.ndarray, "*batch 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    *,
    rest_joints: Float[np.ndarray, "*batch 24 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    joint_transforms: Float[np.ndarray, "*batch 24 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
):
    return _forward_vertices(
        skin_weights=weights.skin_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rest_joints=rest_joints,
        rest_vertices=rest_vertices,
        joint_transforms=joint_transforms,
        pose_offsets=pose_offsets,
        xp=np,
    )


def forward_skeleton(
    weights: SkelWeights,
    global_rotation: Float[np.ndarray, "*batch 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    *,
    joint_transforms: Float[np.ndarray, "*batch 24 4 4"],
):
    return _forward_skeleton(
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        joint_transforms=joint_transforms,
        xp=np,
    )
