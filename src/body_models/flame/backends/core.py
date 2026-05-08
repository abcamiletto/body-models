"""Backend-agnostic FLAME computation."""

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
    shapedirs: Float[Array, "V 3 N_shape"],
    exprdirs: Float[Array, "V 3 N_expr"],
    posedirs: Float[Array, "P V*3"],
    lbs_weights: Float[Array, "V 5"],
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 N_shape"],
    j_exprdirs: Float[Array, "5 3 N_expr"],
    parents: list[int],
    kinematic_fronts: list[Front],
    # Inputs
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 4 N"] | Float[Array, "B 4 3 3"],
    head_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert expression.ndim == 2 and expression.shape[1] >= 1
    assert pose.ndim in (3, 4) and pose.shape[1] == 4
    assert global_translation is None or (global_translation.ndim == 2 and global_translation.shape[1] == 3)

    if xp is None:
        xp = get_namespace(shape)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        v_template = v_template[vertex_indices]
        shapedirs = shapedirs[vertex_indices]
        exprdirs = exprdirs[vertex_indices]
        lbs_weights = lbs_weights[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)

    v_t, j_t, pose_matrices, T_world = _forward_core(
        xp=xp,
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        j_exprdirs=j_exprdirs,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        shape=shape,
        expression=expression,
        pose=pose,
        head_rotation=head_rotation,
        skeleton_only=False,
        rotation_type=rotation_type,
    )
    assert v_t is not None  # guaranteed when skeleton_only=False

    eye3 = common.eye_as(pose_matrices, batch_dims=(pose.shape[0], 1), xp=xp)
    pose_delta = (pose_matrices[:, 1:] - eye3).reshape(pose.shape[0], -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(pose.shape[0], -1, 3)
    v_posed = smpl_core.linear_blend_skinning(xp, v_shaped, j_t, T_world, lbs_weights)
    v_posed = smpl_core.apply_global_transform(xp, v_posed, global_rotation, global_translation, rotation_type)

    return v_posed


def forward_skeleton(
    # Model data
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 N_shape"],
    j_exprdirs: Float[Array, "5 3 N_expr"],
    parents: list[int],
    kinematic_fronts: list[Front],
    # Inputs
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 4 N"] | Float[Array, "B 4 3 3"],
    head_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any = None,
) -> Float[Array, "B 5 4 4"]:
    """Compute skeleton joint transforms [B, 5, 4, 4]."""
    assert shape.ndim == 2 and shape.shape[1] >= 1
    assert expression.ndim == 2 and expression.shape[1] >= 1
    assert pose.ndim in (3, 4) and pose.shape[1] == 4
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
        exprdirs=None,
        j_template=j_template,
        j_shapedirs=j_shapedirs,
        j_exprdirs=j_exprdirs,
        parents=parents,
        kinematic_fronts=active_fronts,
        shape=shape,
        expression=expression,
        pose=pose,
        head_rotation=head_rotation,
        skeleton_only=True,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
    )

    # Extract R and t from T_world
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]

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
    shapedirs: Float[Array, "V 3 N_shape"] | None,
    exprdirs: Float[Array, "V 3 N_expr"] | None,
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 N_shape"],
    j_exprdirs: Float[Array, "5 3 N_expr"],
    parents: list[int],
    kinematic_fronts: list[Front],
    shape: Float[Array, "B N_shape"],
    expression: Float[Array, "B N_expr"],
    pose: Float[Array, "B 4 N"] | Float[Array, "B 4 3 3"],
    head_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    skeleton_only: bool,
    rotation_type: RotationType,
    joint_indices: list[int] | None = None,
) -> tuple[
    Float[Array, "B V 3"] | None,
    Float[Array, "B 5 3"],
    Float[Array, "B 5 3 3"],
    Float[Array, "B 5 4 4"],
]:
    """Core forward pass."""
    B = pose.shape[0]

    # Broadcast shape if needed
    if shape.shape[0] == 1 and B > 1:
        shape = xp.broadcast_to(shape, (B, shape.shape[1]))

    pose_matrices = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    if head_rotation is None:
        root_matrices = SO3.identity_as(
            pose_matrices,
            batch_dims=(B, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        root_matrices = SO3.convert(head_rotation, src=rotation_type, dst="rotmat", xp=xp)[:, None]
    pose_matrices = xp.concat([root_matrices, pose_matrices], axis=1)

    shape_dim = min(shape.shape[-1], j_shapedirs.shape[-1])
    expr_dim = min(expression.shape[-1], j_exprdirs.shape[-1])
    params = xp.concat([shape[:, :shape_dim], expression[:, :expr_dim]], axis=-1)
    j_dirs = xp.concat([j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expr_dim]], axis=-1)
    j_t = j_template + xp.einsum("bi,jdi->bjd", params, j_dirs)

    # Shape blend shapes for mesh output
    if skeleton_only:
        v_t = None
    else:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        dirs_simp = xp.concat([shapedirs[:, :, :shape_dim], exprdirs[:, :, :expr_dim]], axis=-1)
        v_t = v_template + xp.einsum("bi,vdi->bvd", params, dirs_simp)

    # Forward kinematics
    # Build t_local using concatenation
    j0 = j_t[:, 0:1]  # [B, 1, 3]
    j_rest = j_t[:, 1:] - j_t[:, parents[1:]]  # [B, J-1, 3]
    t_local = xp.concat([j0, j_rest], axis=1)  # [B, J, 3]

    T_world = smpl_core.batched_forward_kinematics(xp, pose_matrices, t_local, kinematic_fronts, joint_indices)

    return v_t, j_t, pose_matrices, T_world
