"""SciPy sparse SMPL backend."""

import numpy as np
from jaxtyping import Float
from scipy import sparse

from body_models.rotations import RotationType
from body_models.smpl.backends import core
from body_models.smpl.io import SmplWeights

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
    skip_vertices: bool = False,
) -> core.SmplPreparedPose:
    return core.prepare_pose(
        xp=np,
        posedirs=weights.posedirs,
        kinematic_fronts=weights.kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        skip_vertices=skip_vertices,
    )


def sparse_skin(
    v_shaped,
    j_t,
    T_world,
    lbs_weights,
    *,
    global_rotation=None,
    global_translation=None,
    rotation_type: RotationType = "axis_angle",
):
    lbs_weights = sparse.csr_array(lbs_weights)
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    t_adjusted = t_world - np.squeeze(R_world @ j_t[..., None], axis=-1)

    v_posed = np.empty_like(v_shaped)
    num_joints = j_t.shape[-2]
    for batch in np.ndindex(v_shaped.shape[:-2]):
        W_R = lbs_weights @ R_world[batch].reshape(num_joints, 9)
        W_t = lbs_weights @ t_adjusted[batch]
        v_posed[batch] = np.einsum("vij,vj->vi", W_R.reshape(-1, 3, 3), v_shaped[batch]) + W_t

    return core.apply_global_transform(np, v_posed, global_rotation, global_translation, rotation_type)


def forward_vertices(
    weights: SmplWeights,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    joint_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"],
):
    v_shaped = rest_vertices + pose_offsets
    lbs_weights = weights.lbs_weights
    if vertex_indices is not None:
        v_shaped = v_shaped[..., vertex_indices, :]
        lbs_weights = lbs_weights[vertex_indices]

    return sparse_skin(
        v_shaped,
        rest_joints,
        joint_transforms,
        lbs_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )


def forward_skeleton(
    weights: SmplWeights,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"] | None = None,
    joint_transforms: Float[np.ndarray, "*batch J 4 4"],
    pose_offsets: Float[np.ndarray, "*batch V 3"] | None = None,
):
    return core.forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        joint_transforms=joint_transforms,
        xp=np,
    )
