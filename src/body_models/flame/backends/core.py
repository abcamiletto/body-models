"""Backend-agnostic FLAME computation."""

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float

from body_models import common
from body_models.common import get_namespace
from body_models.smpl.backends import core as smpl_core
from nanomanifold import SO3

from body_models.rotations import RotationType

Array = Any  # Generic array type (numpy, torch, jax)
Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


class FlameIdentity(TypedDict):
    """Shape/expression-dependent FLAME state returned by ``prepare_identity``."""

    rest_joints: Float[Array, "*batch J 3"]
    local_joint_offsets: Float[Array, "*batch J 3"]
    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]


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
    pose: Float[Array, "B 4 N"] | Float[Array, "B 4 3 3"],
    head_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Array, "*batch J 3"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_vertices: Float[Array, "*batch V 3"],
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    """Compute mesh vertices [B, V, 3]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(pose)
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        rest_vertices = rest_vertices[..., vertex_indices, :]
        lbs_weights = lbs_weights[vertex_indices]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices].reshape(posedirs.shape[0], -1)
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(pose.shape[:-pose_ndim])

    pose_matrices, T_world = _forward_core(
        xp=xp,
        parents=parents,
        kinematic_fronts=kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        pose=pose,
        head_rotation=head_rotation,
        rotation_type=rotation_type,
    )

    eye3 = common.eye_as(pose_matrices, batch_dims=(*batch_shape, 1), xp=xp)
    pose_delta = (pose_matrices[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
    v_shaped = rest_vertices + (pose_delta @ posedirs).reshape(*batch_shape, -1, 3)
    v_posed = smpl_core.linear_blend_skinning(xp, v_shaped, rest_joints, T_world, lbs_weights)
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
    pose: Float[Array, "B 4 N"] | Float[Array, "B 4 3 3"],
    head_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[Array, "*batch J 3"],
    local_joint_offsets: Float[Array, "*batch J 3"],
    rest_vertices: Float[Array, "*batch V 3"] | None = None,
    xp: Any = None,
) -> Float[Array, "B 5 4 4"]:
    """Compute skeleton joint transforms [B, 5, 4, 4]."""
    assert global_translation is None or (global_translation.ndim >= 1 and global_translation.shape[-1] == 3)

    if xp is None:
        xp = get_namespace(pose)
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
    _, T_world = _forward_core(
        xp=xp,
        parents=parents,
        kinematic_fronts=active_fronts,
        local_joint_offsets=local_joint_offsets,
        pose=pose,
        head_rotation=head_rotation,
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
    parents: list[int],
    kinematic_fronts: list[Front],
    local_joint_offsets: Float[Array, "*batch J 3"],
    pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    joint_indices: list[int] | None = None,
) -> tuple[
    Float[Array, "B 5 3 3"],
    Float[Array, "B 5 4 4"],
]:
    """Core forward pass."""
    num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
    pose_ndim = num_rot_dims + 1
    batch_shape = tuple(pose.shape[:-pose_ndim])

    pose_matrices = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    if head_rotation is None:
        root_matrices = SO3.identity_as(
            pose_matrices,
            batch_dims=(*batch_shape, 1),
            rotation_type="rotmat",
            xp=xp,
        )
    else:
        root_matrices = SO3.convert(head_rotation, src=rotation_type, dst="rotmat", xp=xp)[..., None, :, :]
    pose_matrices = xp.concat([root_matrices, pose_matrices], axis=-3)

    T_world = smpl_core.batched_forward_kinematics(
        xp, pose_matrices, local_joint_offsets, kinematic_fronts, joint_indices
    )

    return pose_matrices, T_world


def prepare_identity(
    *,
    xp,
    v_template: Float[Array, "V 3"] | None,
    shapedirs: Float[Array, "V 3 N_shape"] | None,
    exprdirs: Float[Array, "V 3 N_expr"] | None,
    j_template: Float[Array, "5 3"],
    j_shapedirs: Float[Array, "5 3 N_shape"],
    j_exprdirs: Float[Array, "5 3 N_expr"],
    parents: list[int],
    shape: Float[Array, "*batch N_shape"],
    expression: Float[Array, "*batch N_expr"],
    skip_vertices: bool = False,
) -> FlameIdentity:
    """Precompute shape/expression-dependent FLAME state for repeated forward passes."""
    assert shape.ndim >= 1 and shape.shape[-1] >= 1
    assert expression.ndim >= 1 and expression.shape[-1] >= 1

    shape_dim = shape.shape[-1]
    expr_dim = expression.shape[-1]
    params = xp.concat([shape, expression], axis=-1)
    j_dirs = xp.concat([j_shapedirs[:, :, :shape_dim], j_exprdirs[:, :, :expr_dim]], axis=-1)
    rest_joints = j_template + xp.einsum("...i,jdi->...jd", params, j_dirs)
    local_joint_offsets = xp.concat(
        [rest_joints[..., 0:1, :], rest_joints[..., 1:, :] - rest_joints[..., parents[1:], :]], axis=-2
    )
    identity: FlameIdentity = {
        "rest_joints": rest_joints,
        "local_joint_offsets": local_joint_offsets,
    }
    if not skip_vertices:
        assert v_template is not None and shapedirs is not None and exprdirs is not None
        dirs_simp = xp.concat([shapedirs[:, :, :shape_dim], exprdirs[:, :, :expr_dim]], axis=-1)
        identity["rest_vertices"] = v_template + xp.einsum("...i,vdi->...vd", params, dirs_simp)
    return identity
