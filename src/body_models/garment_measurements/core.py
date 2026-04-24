"""Backend-agnostic GarmentMeasurements body model computation."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float
from nanomanifold import SE3, SO3

from .. import common
from ..rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]


def forward_vertices(
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    bind_quats: Float[Array, "J 4"],
    skin_weights: Float[Array, "V J"],
    mvc_weights: Float[Array, "V J"],
    kinematic_fronts: list[Front],
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
    batch_size = _input_batch_size(shape, pose, global_rotation, global_translation)
    shape = _broadcast_batch(shape, batch_size, xp=xp)
    shaped_vertices = _shape_vertices(mean_vertices, components, eigenvalues, shape, xp=xp)
    bind_skeleton, posed_skeleton = _forward_skeleton_se3(
        shaped_vertices=shaped_vertices,
        bind_quats=bind_quats,
        mvc_weights=mvc_weights,
        kinematic_fronts=kinematic_fronts,
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
    kinematic_fronts: list[Front],
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
    batch_size = _input_batch_size(shape, pose, global_rotation, global_translation)
    shape = _broadcast_batch(shape, batch_size, xp=xp)
    shaped_vertices = _shape_vertices(mean_vertices, components, eigenvalues, shape, xp=xp)
    _, skeleton = _forward_skeleton_se3(
        shaped_vertices=shaped_vertices,
        bind_quats=bind_quats,
        mvc_weights=mvc_weights,
        kinematic_fronts=kinematic_fronts,
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
    kinematic_fronts: list[Front],
    pose: Float[Array, "B J N"] | Float[Array, "B J 3 3"] | None,
    rotation_type: RotationType,
    xp: Any,
) -> tuple[Float[Array, "B J 7"], Float[Array, "B J 7"]]:
    batch_size = shaped_vertices.shape[0]
    joint_positions = xp.einsum("vj,bvd->bjd", _match_dtype(mvc_weights, shaped_vertices, xp=xp), shaped_vertices)
    bind_quats = xp.broadcast_to(_match_dtype(bind_quats, shaped_vertices, xp=xp), (batch_size, *bind_quats.shape))
    bind_global_quats = _propagate_quats(bind_quats, kinematic_fronts, xp=xp)
    bind_trans = _local_translations_from_positions(joint_positions, bind_global_quats, kinematic_fronts, xp=xp)

    bind_local = SE3.from_rt(bind_quats, bind_trans, xp=xp)
    bind_global = _propagate_se3(bind_local, kinematic_fronts, xp=xp)

    pose_quats = _pose_quats(pose, batch_size, bind_quats.shape[1], rotation_type, bind_quats, xp=xp)
    posed_quats = SO3.multiply(bind_quats, pose_quats, xp=xp)
    posed_local = SE3.from_rt(posed_quats, bind_trans, xp=xp)
    posed_global = _propagate_se3(posed_local, kinematic_fronts, xp=xp)
    return bind_global, posed_global


def _local_translations_from_positions(
    positions: Float[Array, "B J 3"],
    bind_global_quats: Float[Array, "B J 4"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "B J 3"]:
    translations = xp.zeros_like(positions)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            translations = common.set(translations, (slice(None), joints), positions[:, joints], copy=False, xp=xp)
            continue

        offset = positions[:, joints] - positions[:, parents]
        parent_inv = SO3.inverse(bind_global_quats[:, parents], xp=xp)
        local_translations = SO3.rotate_points(parent_inv, offset[..., None, :], xp=xp).squeeze(-2)
        translations = common.set(translations, (slice(None), joints), local_translations, copy=False, xp=xp)
    return translations


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


def _propagate_quats(
    quats: Float[Array, "B J 4"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "B J 4"]:
    globals_ = xp.zeros_like(quats)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            front = quats[:, joints]
        else:
            front = SO3.multiply(globals_[:, parents], quats[:, joints], xp=xp)
        globals_ = common.set(globals_, (slice(None), joints), front, copy=False, xp=xp)
    return globals_


def _propagate_se3(
    se3: Float[Array, "B J 7"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "B J 7"]:
    globals_ = xp.zeros_like(se3)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            front = se3[:, joints]
        else:
            front = SE3.multiply(globals_[:, parents], se3[:, joints], xp=xp)
        globals_ = common.set(globals_, (slice(None), joints), front, copy=False, xp=xp)
    return globals_


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
        quat = _broadcast_batch(quat, batch_size, xp=xp)

    if global_translation is None:
        translation = common.zeros_as(ref, shape=(batch_size, 3), xp=xp)
    else:
        translation = _match_dtype(global_translation, ref, xp=xp)
        translation = _broadcast_batch(translation, batch_size, xp=xp)
    return quat, translation


def _input_batch_size(*values: Array | None) -> int:
    sizes = [value.shape[0] for value in values if value is not None]
    active_sizes = {size for size in sizes if size != 1}
    if len(active_sizes) > 1:
        raise ValueError(f"Inputs must have compatible batch sizes, got {sizes}")
    return next(iter(active_sizes), 1)


def _broadcast_batch(value: Array, batch_size: int, *, xp: Any) -> Array:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(f"Input batch size {value.shape[0]} cannot broadcast to {batch_size}")
    return xp.broadcast_to(value, (batch_size, *value.shape[1:]))


def _match_dtype(value: Array, ref: Array, *, xp: Any) -> Array:
    zero = xp.zeros_like(ref.reshape(-1)[:1])
    return value + zero
