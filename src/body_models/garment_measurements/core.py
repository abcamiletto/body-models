"""Backend-agnostic GarmentMeasurements body model computation."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float
from nanomanifold import SE3, SO3

from .. import common
from ..rotations import RotationType

Array = Any


def forward_vertices(
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    bind_quats: Float[Array, "J 4"],
    skin_weights: Float[Array, "V J"],
    mvc_weights: Float[Array, "V J"],
    parents: list[int],
    shape: Float[Array, "B C"],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    """Evaluate shaped and posed body vertices [B, V, 3]."""
    shaped_vertices = _shape_vertices(mean_vertices, components, eigenvalues, shape, xp=xp)
    bind_skeleton, posed_skeleton = _forward_skeleton_se3(
        shaped_vertices=shaped_vertices,
        bind_quats=bind_quats,
        mvc_weights=mvc_weights,
        parents=parents,
        pose=pose,
        rotation_type=rotation_type,
        xp=xp,
    )

    final_skeleton = SE3.multiply(posed_skeleton, SE3.inverse(bind_skeleton, xp=xp), xp=xp)

    vertices = shaped_vertices
    weights = skin_weights
    if vertex_indices is not None:
        vertex_indices = xp.asarray(vertex_indices)
        vertices = vertices[:, vertex_indices]
        weights = weights[vertex_indices]

    skin_mats = SE3.to_matrix(final_skeleton, xp=xp)
    skin_rot = skin_mats[:, :, :3, :3]
    skin_trans = skin_mats[:, :, :3, 3]
    weighted_rot = xp.einsum("vj,bjkl->bvkl", weights, skin_rot)
    weighted_trans = xp.einsum("vj,bjk->bvk", weights, skin_trans)
    vertices = xp.squeeze(weighted_rot @ vertices[..., None], axis=-1) + weighted_trans

    return _apply_global_transform(
        values=vertices,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
        xp=xp,
    )


def forward_skeleton(
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    bind_quats: Float[Array, "J 4"],
    mvc_weights: Float[Array, "V J"],
    parents: list[int],
    shape: Float[Array, "B C"],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"] | None = None,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
    global_translation: Float[Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    xp: Any,
) -> Float[Array, "B J 4 4"]:
    """Compute world-space joint transforms [B, J, 4, 4]."""
    shaped_vertices = _shape_vertices(mean_vertices, components, eigenvalues, shape, xp=xp)
    _, skeleton = _forward_skeleton_se3(
        shaped_vertices=shaped_vertices,
        bind_quats=bind_quats,
        mvc_weights=mvc_weights,
        parents=parents,
        pose=pose,
        rotation_type=rotation_type,
        xp=xp,
    )

    skeleton = _apply_global_se3(
        skeleton=skeleton,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
        xp=xp,
    )

    if joint_indices is not None:
        skeleton = skeleton[:, xp.asarray(joint_indices)]

    return SE3.to_matrix(skeleton, xp=xp)


def _shape_vertices(
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    shape: Float[Array, "B C"],
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    assert shape.ndim == 2
    scaled_shape = shape * xp.sqrt(_match_dtype(eigenvalues, shape, xp=xp))[None]
    return _match_dtype(mean_vertices, shape, xp=xp)[None] + xp.einsum(
        "bc,vdc->bvd",
        scaled_shape,
        _match_dtype(components, shape, xp=xp),
    )


def _forward_skeleton_se3(
    *,
    shaped_vertices: Float[Array, "B V 3"],
    bind_quats: Float[Array, "J 4"],
    mvc_weights: Float[Array, "V J"],
    parents: list[int],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"] | None,
    rotation_type: RotationType,
    xp: Any,
) -> tuple[Float[Array, "B J 7"], Float[Array, "B J 7"]]:
    batch_size = shaped_vertices.shape[0]
    joint_positions = xp.einsum("vj,bvd->bjd", _match_dtype(mvc_weights, shaped_vertices, xp=xp), shaped_vertices)
    bind_quats = xp.broadcast_to(_match_dtype(bind_quats, shaped_vertices, xp=xp), (batch_size, *bind_quats.shape))
    bind_global_quats = _propagate_quats(bind_quats, parents, xp=xp)
    bind_trans = _local_translations_from_positions(joint_positions, bind_global_quats, parents, xp=xp)

    bind_local = SE3.from_rt(bind_quats, bind_trans, xp=xp)
    bind_global = _propagate_se3(bind_local, parents, xp=xp)

    pose_quats = _pose_quats(pose, batch_size, bind_quats.shape[1], rotation_type, bind_quats, xp=xp)
    posed_quats = SO3.multiply(bind_quats, pose_quats, xp=xp)
    posed_local = SE3.from_rt(posed_quats, bind_trans, xp=xp)
    posed_global = _propagate_se3(posed_local, parents, xp=xp)
    return bind_global, posed_global


def _local_translations_from_positions(
    positions: Float[Array, "B J 3"],
    bind_global_quats: Float[Array, "B J 4"],
    parents: list[int],
    *,
    xp: Any,
) -> Float[Array, "B J 3"]:
    translations = []
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            translations.append(positions[:, joint_index])
            continue
        offset = positions[:, joint_index] - positions[:, parent_index]
        parent_inv = SO3.inverse(bind_global_quats[:, parent_index], xp=xp)
        translations.append(SO3.rotate_points(parent_inv, offset[:, None], xp=xp).squeeze(-2))
    return xp.stack(translations, axis=1)


def _pose_quats(
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"] | None,
    batch_size: int,
    num_joints: int,
    rotation_type: RotationType,
    ref: Float[Array, "B J 4"],
    *,
    xp: Any,
) -> Float[Array, "B J 4"]:
    if pose is None:
        return SO3.identity_as(ref, batch_dims=(batch_size, num_joints), rotation_type="quat", xp=xp)
    pose = _match_dtype(pose, ref, xp=xp)
    if pose.shape[0] == 1 and batch_size > 1:
        pose = xp.broadcast_to(pose, (batch_size, *pose.shape[1:]))
    return SO3.convert(pose, src=rotation_type, dst="quat", xp=xp)


def _propagate_quats(quats: Float[Array, "B J 4"], parents: list[int], *, xp: Any) -> Float[Array, "B J 4"]:
    globals_ = []
    for joint_index, parent_index in enumerate(parents):
        quat = quats[:, joint_index]
        if parent_index >= 0:
            quat = SO3.multiply(globals_[parent_index], quat, xp=xp)
        globals_.append(quat)
    return xp.stack(globals_, axis=1)


def _propagate_se3(se3: Float[Array, "B J 7"], parents: list[int], *, xp: Any) -> Float[Array, "B J 7"]:
    globals_ = []
    for joint_index, parent_index in enumerate(parents):
        transform = se3[:, joint_index]
        if parent_index >= 0:
            transform = SE3.multiply(globals_[parent_index], transform, xp=xp)
        globals_.append(transform)
    return xp.stack(globals_, axis=1)


def _apply_global_transform(
    *,
    values: Float[Array, "B N 3"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    global_translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
    xp: Any,
) -> Float[Array, "B N 3"]:
    if global_rotation is None and global_translation is None:
        return values
    quat, translation = _global_quat_translation(
        xp=xp,
        ref=values,
        batch_size=values.shape[0],
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
    return SE3.transform_points(SE3.from_rt(quat, translation, xp=xp), values, xp=xp)


def _apply_global_se3(
    *,
    skeleton: Float[Array, "B J 7"],
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    global_translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
    xp: Any,
) -> Float[Array, "B J 7"]:
    if global_rotation is None and global_translation is None:
        return skeleton
    quat, translation = _global_quat_translation(
        xp=xp,
        ref=skeleton,
        batch_size=skeleton.shape[0],
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )
    return SE3.multiply(SE3.from_rt(quat[:, None], translation[:, None], xp=xp), skeleton, xp=xp)


def _global_quat_translation(
    *,
    xp: Any,
    ref: Float[Array, "..."],
    batch_size: int,
    global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None,
    global_translation: Float[Array, "B 3"] | None,
    rotation_type: RotationType,
) -> tuple[Float[Array, "B 4"], Float[Array, "B 3"]]:
    if global_rotation is None:
        quat = SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type="quat", xp=xp)
    else:
        quat = SO3.convert(_match_dtype(global_rotation, ref, xp=xp), src=rotation_type, dst="quat", xp=xp)

    if global_translation is None:
        translation = common.zeros_as(ref, shape=(batch_size, 3), xp=xp)
    else:
        translation = _match_dtype(global_translation, ref, xp=xp)
    return quat, translation


def _match_dtype(value: Array, ref: Array, *, xp: Any) -> Array:
    zero = xp.zeros_like(ref.reshape(-1)[:1])
    return value + zero
