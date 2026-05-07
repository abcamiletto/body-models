"""Backend-agnostic MANO computation."""

from typing import Any

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from body_models.smpl.backends import core as smpl_core
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V D 10"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 16"],
    j_template: Float[Array, "16 3"],
    j_shapedirs: Float[Array, "16 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    # Inputs
    shape: Float[Array, "B 10"],
    hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
    wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        v_template = v_template[vertex_indices]
        shapedirs = shapedirs[vertex_indices]
        lbs_weights = lbs_weights[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        shape=shape,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )

    eye3 = common.eye_as(pose_matrices, batch_dims=(hand_pose.shape[0], 1), xp=xp)
    pose_delta = (pose_matrices[:, 1:] - eye3).reshape(hand_pose.shape[0], -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(hand_pose.shape[0], -1, 3)
    v_posed = smpl_core.linear_blend_skinning(xp, v_shaped, j_t, T_world, lbs_weights)
    v_posed = smpl_core.apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    # Inputs
    shape: Float[Array, "B 10"],
    hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
    wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    active_fronts = kinematic_fronts
    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        if any(joint < 0 or joint >= len(parents) for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {len(parents)})")

        active_joints = set()
        for joint in joint_indices:
            cur = joint
            while cur >= 0 and cur not in active_joints:
                active_joints.add(cur)
                cur = parents[cur]

        active_fronts = []
        for joints, joint_parents in kinematic_fronts:
            pairs = [(joint, parent) for joint, parent in zip(joints, joint_parents) if joint in active_joints]
            if pairs:
                active_fronts.append(([joint for joint, _ in pairs], [parent for _, parent in pairs]))

    _, _, _, T_world = _forward_core(
        xp=xp,
        v_template=None,
        shapedirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        parents=parents,
        kinematic_fronts=active_fronts,
        hand_mean=hand_mean,
        shape=shape,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        skeleton_only=True,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
    )

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
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V D 10"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "45"],
    shape: Float[Array, "B 10"],
    hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
    wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    skeleton_only: bool,
    rotation_type: RotationType,
    joint_indices: list[int] | None = None,
) -> tuple[
    Float[Array, "B V 3"] | None,
    Float[Array, "B J 3"],
    Float[Array, "B J 3 3"],
    Float[Array, "B J 4 4"],
]:
    """Core forward pass."""
    B = hand_pose.shape[0]

    # Broadcast shape if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

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
            batch_dims=(B, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        wrist_matrices = SO3.convert(
            wrist_rotation,
            src=rotation_type,
            dst="rotmat",
            xp=xp,
        )[:, None]
    pose_matrices = xp.concat([wrist_matrices, hand_pose_matrices], axis=1)

    # Joint locations from precomputed regression matrices
    j_t = j_template + xp.einsum("...p,jdp->...jd", shape, j_shapedirs[:, :, : shape.shape[-1]])

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None
        v_t = v_template + xp.einsum("bi,vdi->bvd", shape, shapedirs[:, :, : shape.shape[-1]])

    # Forward kinematics
    # Build t_local using concatenation
    j0 = j_t[:, 0:1]  # [B, 1, 3]
    j_rest = j_t[:, 1:] - j_t[:, parents[1:]]  # [B, J-1, 3]
    t_local = xp.concat([j0, j_rest], axis=1)  # [B, J, 3]

    T_world = smpl_core.batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts, joint_indices)

    return v_t, j_t, pose_matrices, T_world
