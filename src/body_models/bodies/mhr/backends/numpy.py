"""NumPy MHR backend."""

import numpy as np
from jaxtyping import Float

from body_models.bodies.mhr.backends.core import MhrIdentity, MhrPreparedPose
from body_models.bodies.mhr.backends.core import forward_skeleton as _forward_skeleton
from body_models.bodies.mhr.backends.core import forward_vertices as _forward_vertices
from body_models.bodies.mhr.backends.core import prepare_identity as _prepare_identity
from body_models.bodies.mhr.backends.core import prepare_pose as _prepare_pose
from body_models.bodies.mhr.io import MhrWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: MhrWeights,
    shape: Float[np.ndarray, "*batch 45"],
    expression: Float[np.ndarray, "*batch 72"],
    skip_vertices: bool = False,
) -> MhrIdentity:
    return _prepare_identity(
        xp=np,
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        shape=shape,
        expression=expression,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: MhrWeights, pose: Float[np.ndarray, "*batch 204"], skip_vertices: bool = False
) -> MhrPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=np,
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        bind_inv_linear=weights.bind_inv_linear,
        bind_inv_translation=weights.bind_inv_translation,
        corrective_W1=weights.corrective_W1,
        corrective_W2=weights.corrective_W2,
        pose=pose,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: MhrWeights,
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    skinning_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
):
    return _forward_vertices(
        joint_weights=weights.skin_weights,
        joint_indices=weights.skin_indices,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rest_vertices=rest_vertices,
        skinning_transforms=skinning_transforms,
        pose_offsets=pose_offsets,
        xp=np,
    )


def forward_skeleton(
    weights: MhrWeights,
    *,
    skeleton_transforms: Float[np.ndarray, "*batch J 4 4"],
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
):
    return _forward_skeleton(
        num_joints=len(weights.parents),
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        skeleton_transforms=skeleton_transforms,
        xp=np,
    )
