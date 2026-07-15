"""Backend-agnostic SMPL computation."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


class SmplIdentity(TypedDict):
    """Shape-dependent SMPL state returned by ``prepare_identity``."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]
    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]


class SmplPreparedPose(TypedDict):
    """Pose-dependent SMPL state returned by ``prepare_pose``."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: NotRequired[Float[Array, "*batch V 3"]]


def forward_vertices(
    # Model data
    lbs_weights: Float[Array, "V 24"],
    rest_vertices: Float[Array, "*batch V 3"],
    skinning_transforms: Float[Array, "*batch J 4 4"],
    pose_offsets: Float[Array, "*batch V 3"],
    global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_translation: Float[Array, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(rest_vertices)

    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        lbs_weights = lbs_weights[vertex_indices]
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]

    v_shaped = rest_vertices + pose_offsets
    v_posed = linear_blend_skinning(xp, v_shaped, skinning_transforms, lbs_weights)
    v_posed = apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def prepare_pose(
    # Model data
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    skip_vertices: bool = False,
    xp: Any = None,
) -> SmplPreparedPose:
    """Precompute pose-dependent SMPL state for repeated forward passes."""
    if xp is None:
        xp = get_namespace(body_pose)
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    local_joint_offsets = xp.broadcast_to(local_joint_offsets, (*batch_shape, *local_joint_offsets.shape[-2:]))
    rest_joints = xp.broadcast_to(rest_joints, (*batch_shape, *rest_joints.shape[-2:]))

    pose_matrices, T_world = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
    )

    pose: SmplPreparedPose = {
        "skeleton_transforms": T_world,
        "skinning_transforms": bind_relative_transforms(xp, T_world, rest_joints),
    }
    if skip_vertices:
        return pose
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=xp)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    pose["pose_offsets"] = (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    return pose


def forward_skeleton(
    parents: list[int],
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_translation: Float[Array, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "*batch J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(skeleton_transforms)
    T_world = skeleton_transforms
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        if any(joint < 0 or joint >= len(parents) for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {len(parents)})")
        T_world = T_world[..., joint_indices, :, :]

    if global_rotation is None and global_translation is None:
        return T_world

    if global_rotation is None:
        assert global_translation is not None
        t = T_world[..., :3, 3] + global_translation[..., None, :]
        return common.set(T_world, (..., slice(None, 3), 3), t, xp=xp)

    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    R_global = SO3.convert(global_rotation, src=rotation_type, dst="rotmat", xp=xp)
    t_world = (R_global @ t_world.mT).mT
    R_world = R_global[..., None, :, :] @ R_world
    if global_translation is not None:
        t_world = t_world + global_translation[..., None, :]

    batch_shape = R_world.shape[:-3]
    J = R_world.shape[-3]
    upper = xp.concat([R_world, t_world[..., None]], axis=-1)
    bottom = common.zeros_as(upper, shape=(*batch_shape, J, 1, 4), xp=xp)
    bottom = common.set(bottom, (..., 0, 3), 1.0, xp=xp)
    return xp.concat([upper, bottom], axis=-2)


def _forward_core(
    xp,
    kinematic_fronts: list[Front],
    body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    local_joint_offsets: Float[Array, "*batch J 3"],
) -> tuple[
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch J 4 4"],
]:
    """Core forward pass."""
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])

    # Build full pose with pelvis rotation
    body_pose_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        pelvis_matrices = SO3.convert(
            pelvis_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    pose_matrices = xp.concat([pelvis_matrices, body_pose_matrices], axis=-3)

    T_world = batched_forward_kinematics(xp, pose_matrices, local_joint_offsets, kinematic_fronts)

    return pose_matrices, T_world


def prepare_identity(
    *,
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V D 10"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    skip_vertices: bool = False,
) -> SmplIdentity:
    """Precompute shape-dependent SMPL state for repeated forward passes."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    j_t = j_template + xp.einsum("...p,jdp->...jd", shape, j_shapedirs[:, :, : shape.shape[-1]])
    j0 = j_t[..., 0:1, :]
    j_rest = j_t[..., 1:, :] - j_t[..., parents[1:], :]
    identity: SmplIdentity = {
        "rest_joints": j_t,
        "local_joint_offsets": xp.concat([j0, j_rest], axis=-2),
    }
    if not skip_vertices:
        assert v_template is not None and shapedirs is not None
        identity["rest_vertices"] = v_template + xp.einsum("...i,vdi->...vd", shape, shapedirs[:, :, : shape.shape[-1]])
    return identity


def batched_forward_kinematics(
    xp,
    R: Float[Array, "*batch J 3 3"],
    t: Float[Array, "*batch J 3"],
    fronts: list[Front],
    joint_indices: list[int] | None = None,
) -> Float[Array, "*batch J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts.

    Uses unified 4x4 homogeneous transforms: one bmm per depth level instead
    of two (R_parent @ R_local + R_parent @ t_local).
    """
    J = R.shape[-3]

    batch_shape = R.shape[:-3]
    upper = xp.concat([R, t[..., None]], axis=-1)
    bottom = common.zeros_as(upper, shape=(*batch_shape, J, 1, 4), xp=xp)
    bottom = common.set(bottom, (..., 0, 3), 1.0, xp=xp)
    T_local = xp.concat([upper, bottom], axis=-2)

    T_world: list[Float[Array, "*batch 4 4"] | None] = [None] * J

    for joints, parents in fronts:
        if parents[0] < 0:  # Root joints
            for joint in joints:
                T_world[joint] = T_local[..., joint, :, :]
            continue

        T_parent = xp.stack([T_world[i] for i in parents], axis=-3)
        T_cur = T_parent @ T_local[..., joints, :, :]
        for idx, joint in enumerate(joints):
            T_world[joint] = T_cur[..., idx, :, :]

    if joint_indices is None:
        return xp.stack(T_world, axis=-3)
    return xp.stack([T_world[j] for j in joint_indices], axis=-3)


def linear_blend_skinning(
    xp,
    vertices: Float[Array, "*batch V 3"],
    transforms: Float[Array, "*batch J 4 4"],
    lbs_weights: Float[Array, "V J"],
) -> Float[Array, "*batch V 3"]:
    """Apply linear blend skinning to posed vertices."""
    R = transforms[..., :3, :3]
    t = transforms[..., :3, 3]
    W_R = xp.einsum("vj,...jkl->...vkl", lbs_weights, R)
    W_t = xp.einsum("vj,...jk->...vk", lbs_weights, t)
    rotated = xp.squeeze(W_R @ vertices[..., None], axis=-1)
    return rotated + W_t


def bind_relative_transforms(
    xp,
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    rest_joints: Float[Array, "*batch J 3"],
) -> Float[Array, "*batch J 4 4"]:
    R = skeleton_transforms[..., :3, :3]
    t = skeleton_transforms[..., :3, 3]
    bind_t = t - xp.squeeze(R @ rest_joints[..., None], axis=-1)
    return common.set(skeleton_transforms, (..., slice(None, 3), 3), bind_t, xp=xp)


def apply_global_transform(
    xp,
    points: Float[Array, "*batch N 3"],
    rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    translation: Float[Array, "*batch 3"] | None,
    rotation_type: RotationType,
) -> Float[Array, "*batch N 3"]:
    """Apply global rotation and translation to points."""
    if rotation is not None:
        R = SO3.convert(rotation, src=rotation_type, dst="rotmat", xp=xp)
        points = (R @ points.mT).mT
    if translation is not None:
        points = points + translation[..., None, :]
    return points
