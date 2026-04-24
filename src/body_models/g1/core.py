"""Backend-agnostic Unitree G1 rigid articulated model computation."""

from __future__ import annotations

from typing import Any, Literal

from array_api_compat import get_namespace
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common
from ..rotations import RotationType as SO3RotationType

Array = Any
RotationType = SO3RotationType | Literal["hinge"]

MUJOCO_TO_KIMODO = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
SKIN_WEIGHTS_ERROR = "G1 is a rigid articulated model and does not define skin_weights."
VALID_ROTATION_TYPES = ("axis_angle", "quat", "sixd", "matrix", "rotmat", "hinge")


def forward_skeleton(
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    joint_rotation_axes: Float[Array, "J 3"],
    parents: list[int],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space G1 joint transforms from local rotations."""
    if xp is None:
        xp = get_namespace(pose)
    local_rot = _pose_to_rotmat(pose, joint_rotation_axes, rotation_type=rotation_type, xp=xp)
    batch_size, num_joints = local_rot.shape[:2]
    dtype = local_rot.dtype
    if global_translation is None:
        global_translation = common.zeros_as(local_rot, shape=(batch_size, 3), xp=xp)

    rest_rot = xp.asarray(rest_local_rotations, dtype=dtype)
    local_rot = rest_rot[None] @ local_rot
    local_t = xp.asarray(local_offsets, dtype=dtype)

    rot_world: list[Array | None] = [None] * num_joints
    pos_world: list[Array | None] = [None] * num_joints
    rot_world[0] = local_rot[:, 0]
    pos_world[0] = common.zeros_as(local_rot, shape=(batch_size, 3), xp=xp)
    for joint in range(1, num_joints):
        parent = parents[joint]
        parent_rot = rot_world[parent]
        parent_pos = pos_world[parent]
        rot_world[joint] = parent_rot @ local_rot[:, joint]
        pos_world[joint] = parent_pos + xp.squeeze(parent_rot @ local_t[joint][None, :, None], axis=-1)

    rot = xp.stack(rot_world, axis=1)
    trans = xp.stack(pos_world, axis=1)
    if global_rotation is not None:
        if rotation_type == "hinge":
            global_rot = xp.asarray(global_rotation, dtype=dtype)
        else:
            global_rot = SO3.convert(global_rotation, src=rotation_type, dst="rotmat", xp=xp)
        rot = global_rot[:, None] @ rot
        trans = xp.squeeze(global_rot[:, None] @ trans[..., None], axis=-1)
    trans = trans + global_translation[:, None]

    if joint_indices is not None:
        joint_indices = [int(joint) for joint in joint_indices]
        if any(joint < 0 or joint >= num_joints for joint in joint_indices):
            raise IndexError(f"joint_indices must be in [0, {num_joints})")
        rot = rot[:, joint_indices]
        trans = trans[:, joint_indices]

    last_row = common.zeros_as(rot, shape=(batch_size, rot.shape[1], 1, 4), xp=xp)
    last_row = common.set(last_row, (..., 0, 3), xp.asarray(1.0, dtype=dtype), xp=xp)
    return xp.concat([xp.concat([rot, trans[..., None]], axis=-1), last_row], axis=-2)


def forward_vertices(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    local_offsets: Float[Array, "J 3"],
    rest_local_rotations: Float[Array, "J 3 3"],
    joint_rotation_axes: Float[Array, "J 3"],
    parents: list[int],
    link_joint_indices: list[int],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
    link_geom_positions: Float[Array, "L 3"],
    link_geom_rotations: Float[Array, "L 3 3"],
    link_names: list[str],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    vertex_indices: list[int] | None = None,
    return_per_link: bool = False,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B V 3"] | list[dict[str, Array | str]]:
    """Rigidly transform G1 STL link meshes."""
    if xp is None:
        xp = get_namespace(pose)
    skeleton = forward_skeleton(
        local_offsets=local_offsets,
        rest_local_rotations=rest_local_rotations,
        joint_rotation_axes=joint_rotation_axes,
        parents=parents,
        pose=pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        rotation_type=rotation_type,
        xp=xp,
    )
    joint_rot = skeleton[..., :3, :3]
    joint_pos = skeleton[..., :3, 3]
    source_vertices = xp.asarray(vertices, dtype=pose.dtype)
    geom_pos = xp.asarray(link_geom_positions, dtype=pose.dtype)
    geom_rot = xp.asarray(link_geom_rotations, dtype=pose.dtype)

    if return_per_link:
        per_link = []
    else:
        chunks = []
    for link_idx, joint_idx in enumerate(link_joint_indices):
        start = link_vertex_starts[link_idx]
        count = link_vertex_counts[link_idx]
        local_vertices = source_vertices[start : start + count]
        R = joint_rot[:, joint_idx] @ geom_rot[link_idx]
        t = joint_pos[:, joint_idx] + xp.squeeze(joint_rot[:, joint_idx] @ geom_pos[link_idx][None, :, None], axis=-1)
        transformed = xp.squeeze(R[:, None] @ local_vertices[None, :, :, None], axis=-1) + t[:, None]
        if return_per_link:
            f_start = link_face_starts[link_idx]
            f_count = link_face_counts[link_idx]
            per_link.append(
                {
                    "name": link_names[link_idx],
                    "vertices": transformed,
                    "faces": faces[f_start : f_start + f_count] - start,
                    "joint_index": joint_idx,
                }
            )
        else:
            chunks.append(transformed)

    if return_per_link:
        return per_link
    out = xp.concat(chunks, axis=1)
    if vertex_indices is not None:
        out = out[:, xp.asarray(vertex_indices)]
    return out


def project_pose_to_qpos(
    qpos_joint_indices: list[int],
    qpos_joint_axes: Float[Array, "Q 3"],
    qpos_joint_limits: Float[Array, "Q 2"],
    joint_rotation_axes: Float[Array, "J 3"],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    global_translation: Float[Array, "B 3"] | None = None,
    *,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    clamp_to_limits: bool = True,
    rotation_type: RotationType = "rotmat",
    xp: Any = None,
) -> Float[Array, "B 7+Q"]:
    """Project local 3D rotations to MuJoCo root qpos plus one hinge angle per XML joint."""
    if xp is None:
        xp = get_namespace(pose)
    rot = _pose_to_rotmat(pose, joint_rotation_axes, rotation_type=rotation_type, xp=xp)
    batch_size = rot.shape[0]
    dtype = rot.dtype
    if global_translation is None:
        global_translation = common.zeros_as(rot, shape=(batch_size, 3), xp=xp)
    if global_rotation is None:
        root_rot = rot[:, 0]
    else:
        if rotation_type == "hinge":
            global_rot = xp.asarray(global_rotation, dtype=dtype)
        else:
            global_rot = SO3.convert(global_rotation, src=rotation_type, dst="rotmat", xp=xp)
        root_rot = global_rot @ rot[:, 0]

    coord = xp.asarray(MUJOCO_TO_KIMODO, dtype=dtype)
    kimodo_to_mujoco = coord.mT
    root_t = xp.squeeze(kimodo_to_mujoco @ global_translation[..., None], axis=-1)
    root_rot_mujoco = kimodo_to_mujoco[None] @ root_rot @ coord[None]
    root_quat = SO3.conversions.from_rotmat_to_quat(root_rot_mujoco, convention="wxyz", xp=xp)

    axes = xp.asarray(qpos_joint_axes, dtype=dtype)
    hinge_rots = SO3.convert(rot[:, qpos_joint_indices], src="rotmat", dst="quat", xp=xp)
    angles = SO3.to_hinge(hinge_rots, axes, xp=xp)[..., 0]
    if clamp_to_limits and axes.shape[0] > 0:
        limits = xp.asarray(qpos_joint_limits, dtype=dtype)
        angles = xp.clip(angles, limits[None, :, 0], limits[None, :, 1])
    return xp.concat([root_t, root_quat, angles], axis=1)


def _pose_to_rotmat(
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"],
    joint_rotation_axes: Float[Array, "J 3"],
    *,
    rotation_type: RotationType,
    xp: Any,
) -> Float[Array, "B J 3 3"]:
    if rotation_type == "hinge":
        if pose.ndim != 3 or pose.shape[-1] != 1:
            raise ValueError("G1 hinge poses must have shape [B, J, 1]")
        axes = xp.asarray(joint_rotation_axes, dtype=pose.dtype)
        quat = SO3.from_hinge(pose, axes[None], xp=xp)
        return SO3.convert(quat, src="quat", dst="rotmat", xp=xp)
    if pose.ndim == 3 and pose.shape[-1] == 1:
        raise ValueError('G1 scalar hinge poses require rotation_type="hinge"')
    return SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
