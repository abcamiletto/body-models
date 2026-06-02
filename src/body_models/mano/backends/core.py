"""Backend-agnostic MANO computation."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from body_models.smpl.backends import core as smpl_core
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


class ManoIdentity(TypedDict):
    """Shape-dependent MANO state returned by ``prepare_identity``."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]
    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]


class ManoPreparedPose(TypedDict):
    """Pose-dependent MANO state returned by ``prepare_pose``."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: Float[Array, "*batch J 4 4"]
    pose_offsets: NotRequired[Float[Array, "*batch V 3"]]


def forward_vertices(
    # Model data
    lbs_weights: Float[Array, "V 16"],
    rest_vertices: Float[Array, "*batch V 3"],
    skinning_transforms: Float[Array, "*batch J 4 4"],
    pose_offsets: Float[Array, "*batch V 3"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(rest_vertices)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        rest_vertices = rest_vertices[..., vertex_indices, :]
        pose_offsets = pose_offsets[..., vertex_indices, :]
        lbs_weights = lbs_weights[vertex_indices]

    v_shaped = rest_vertices + pose_offsets
    v_posed = smpl_core.linear_blend_skinning(xp, v_shaped, skinning_transforms, lbs_weights)
    v_posed = smpl_core.apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def prepare_pose(
    # Model data
    posedirs: Float[Array, "P V*3"],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
    wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_joints: Float[Array, "*batch J 3"],
    skip_vertices: bool = False,
    xp: Any = None,
) -> ManoPreparedPose:
    """Precompute pose-dependent MANO state for repeated forward passes."""
    if xp is None:
        xp = get_namespace(hand_pose)
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(hand_pose.shape[:-pose_ndim])
    pose_matrices, T_world = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        local_joint_offsets=local_joint_offsets,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        rotation_type=rotation_type,
    )

    prepared_pose: ManoPreparedPose = {
        "skeleton_transforms": T_world,
        "skinning_transforms": smpl_core.bind_relative_transforms(xp, T_world, rest_joints),
    }
    if skip_vertices:
        return prepared_pose
    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=xp)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    prepared_pose["pose_offsets"] = (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    return prepared_pose


def forward_skeleton(
    parents: list[int],
    skeleton_transforms: Float[Array, "*batch J 4 4"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
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

    # Extract R and t from T_world
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

    # Apply global transform
    if global_rotation is not None or global_translation is not None:
        if global_rotation is not None:
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
    hand_mean: Float[Array, "45"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
    wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
) -> tuple[
    Float[Array, "B J 3 3"],
    Float[Array, "B J 4 4"],
]:
    """Core forward pass."""
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(hand_pose.shape[:-pose_ndim])

    if rotation_type == "axis_angle":
        hand_pose_axis_angle = hand_pose
    else:
        hand_pose_axis_angle = SO3.convert(hand_pose, src=rotation_type, dst="axis_angle", xp=xp)
    hand_pose_axis_angle = hand_pose_axis_angle + hand_mean.reshape(15, 3)

    # Build full pose with wrist rotation
    hand_pose_matrices = SO3.convert(hand_pose_axis_angle, src="axis_angle", dst="rotmat", xp=xp)
    if wrist_rotation is None:
        wrist_matrices = SO3.identity_as(
            hand_pose_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        wrist_matrices = SO3.convert(
            wrist_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[..., None, :, :]
    pose_matrices = xp.concat([wrist_matrices, hand_pose_matrices], axis=-3)

    T_world = smpl_core.batched_forward_kinematics(xp, pose_matrices, local_joint_offsets, kinematic_fronts)

    return pose_matrices, T_world


def prepare_identity(
    *,
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V D 10"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    shape: Float[Array, "*batch 10"],
    skip_vertices: bool = False,
) -> ManoIdentity:
    """Precompute shape-dependent MANO state for repeated forward passes."""
    rest_joints = j_template + xp.einsum("...p,jdp->...jd", shape, j_shapedirs[:, :, : shape.shape[-1]])
    local_joint_offsets = xp.concat(
        [rest_joints[..., 0:1, :], rest_joints[..., 1:, :] - rest_joints[..., parents[1:], :]], axis=-2
    )
    identity: ManoIdentity = {
        "rest_joints": rest_joints,
        "local_joint_offsets": local_joint_offsets,
    }
    if not skip_vertices:
        assert v_template is not None and shapedirs is not None
        identity["rest_vertices"] = v_template + xp.einsum("...i,vdi->...vd", shape, shapedirs[:, :, : shape.shape[-1]])
    return identity
