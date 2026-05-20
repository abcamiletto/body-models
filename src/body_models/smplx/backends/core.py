"""Backend-agnostic SMPL-X computation."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from body_models.smpl.backends import core as smpl_core
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


class SmplxIdentity(TypedDict):
    """Shape- and expression-dependent SMPL-X state returned by ``prepare_identity``."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]
    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]


def forward_vertices(
    # Model data
    v_template: Float[Array, "V 3"],
    shapedirs: Float[Array, "V 3 S"],
    exprdirs: Float[Array, "V 3 E"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 55"],
    j_template: Float[Array, "55 3"],
    j_shapedirs: Float[Array, "55 3 S"],
    j_exprdirs: Float[Array, "55 3 E"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_translation: Float[Array, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Array, "*batch J 3"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_vertices: Float[Array, "*batch V 3"],
    xp: Any = None,
) -> Float[Array, "*batch V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(body_pose)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        rest_vertices = rest_vertices[..., vertex_indices, :]
        lbs_weights = lbs_weights[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    assert tuple(hand_pose.shape[:-pose_ndim]) == batch_shape
    assert tuple(head_pose.shape[:-pose_ndim]) == batch_shape

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        kinematic_fronts=kinematic_fronts,
        hand_mean=hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        rest_vertices=rest_vertices,
    )
    assert v_t is not None

    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=xp)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    v_posed = smpl_core.linear_blend_skinning(xp, v_shaped, j_t, T_world, lbs_weights)
    v_posed = smpl_core.apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: list[int],
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    global_translation: Float[Array, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Array, "*batch J 3"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    xp: Any = None,
) -> Float[Array, "*batch J 4 4"]:
    """Compute skeleton joint transforms [B, J, 4, 4]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(body_pose)
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
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(body_pose.shape[:-pose_ndim])
    assert tuple(hand_pose.shape[:-pose_ndim]) == batch_shape
    assert tuple(head_pose.shape[:-pose_ndim]) == batch_shape
    _, _, _, T_world = _forward_core(
        xp=xp,
        kinematic_fronts=active_fronts,
        hand_mean=hand_mean,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        pelvis_rotation=pelvis_rotation,
        skeleton_only=True,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
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
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    skeleton_only: bool,
    rotation_type: RotationType,
    rest_joints: Float[Array, "*batch J 3"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_vertices: Float[Array, "*batch V 3"] | None = None,
    joint_indices: list[int] | None = None,
) -> tuple[
    Float[Array, "*batch V 3"] | None,
    Float[Array, "*batch J 3"],
    Float[Array, "*batch J 3 3"],
    Float[Array, "*batch J 4 4"],
]:
    """Core forward pass."""
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = body_pose.shape[:-pose_ndim]

    # Apply hand pose mean
    if rotation_type == "axis_angle":
        hand_pose_axis_angle = hand_pose
    else:
        # SMPL-X hand pose mean is defined in axis-angle coordinates.
        hand_pose_axis_angle = SO3.convert(hand_pose, src=rotation_type, dst="axis_angle", xp=xp)
    lh = hand_pose_axis_angle[..., :15, :] + hand_mean[0].reshape(15, 3)
    rh = hand_pose_axis_angle[..., 15:, :] + hand_mean[1].reshape(15, 3)
    hand_pose_adj = xp.concat([lh, rh], axis=-2)

    # Build full pose with pelvis rotation
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        pelvis_matrices = SO3.convert(pelvis_rotation, src=rotation_type, dst="rotmat", xp=xp)[..., None, :, :]
    body_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=xp)
    head_matrices = SO3.convert(head_pose, src=rotation_type, dst="rotmat", xp=xp)
    hand_matrices = SO3.convert(hand_pose_adj, src="axis_angle", dst="rotmat", xp=xp)
    pose_matrices = xp.concat([pelvis_matrices, body_matrices, head_matrices, hand_matrices], axis=-3)

    v_t = None if skeleton_only else rest_vertices

    T_world = smpl_core.batched_forward_kinematics(
        xp, pose_matrices, local_joint_offsets, kinematic_fronts, joint_indices
    )

    return v_t, rest_joints, pose_matrices, T_world


def prepare_identity(
    *,
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V 3 S"] | None,
    exprdirs: Float[Array, "V 3 E"] | None,
    j_template: Float[Array, "J 3"],
    j_shapedirs: Float[Array, "J 3 S"],
    j_exprdirs: Float[Array, "J 3 E"],
    parents: list[int],
    shape: Float[Array, "*batch S"],
    expression: Float[Array, "*batch E"] | None = None,
    skip_vertices: bool = False,
) -> SmplxIdentity:
    """Precompute shape- and expression-dependent SMPL-X state for repeated forward passes."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    batch_shape = shape.shape[:-1]
    if expression is None:
        expression = common.zeros_as(shape, shape=(*batch_shape, 10), xp=xp)
    assert expression.ndim >= 1 and expression.shape[-1] >= 1

    shape_dim = shape.shape[-1]
    expr_dim = expression.shape[-1]
    params = xp.concat([shape, expression], axis=-1)
    j_dirs = xp.concat([j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expr_dim]], axis=-1)
    j_t = j_template + xp.einsum("...p,jdp->...jd", params, j_dirs)
    j0 = j_t[..., 0:1, :]
    j_rest = j_t[..., 1:, :] - j_t[..., parents[1:], :]
    identity: SmplxIdentity = {
        "rest_joints": j_t,
        "local_joint_offsets": xp.concat([j0, j_rest], axis=-2),
    }
    if not skip_vertices:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        dirs = xp.concat([shapedirs[:, :, :shape_dim], exprdirs[:, :, :expr_dim]], axis=-1)
        identity["rest_vertices"] = v_template + xp.einsum("...i,vdi->...vd", params, dirs)
    return identity
