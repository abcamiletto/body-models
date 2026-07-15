"""Backend-independent GarmentMeasurements identity and pose preparation."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from jaxtyping import Float
from nanomanifold import SE3, SO3

from body_models import common
from body_models.rotations import RotationType, rotation_ndim

Array = Any
Front = tuple[list[int], list[int]]


class GarmentMeasurementsIdentity(TypedDict):
    """Shape-dependent GarmentMeasurements state."""

    rest_vertices: NotRequired[Float[Array, "*batch V 3"]]
    bind_skeleton: Float[Array, "*batch J 7"]
    local_bind_translations: Float[Array, "*batch J 3"]


class GarmentMeasurementsPreparedPose(TypedDict):
    """Pose-dependent GarmentMeasurements state."""

    skeleton_transforms: Float[Array, "*batch J 4 4"]
    skinning_transforms: NotRequired[Float[Array, "*batch J 4 4"]]


def prepare_pose(
    bind_quats: Float[Array, "J 4"],
    kinematic_fronts: list[Front],
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    *,
    bind_skeleton: Float[Array, "*batch J 7"],
    local_bind_translations: Float[Array, "*batch J 3"],
    skip_vertices: bool,
    xp: Any,
) -> GarmentMeasurementsPreparedPose:
    """Prepare posed skeleton and bind-relative skinning transforms."""
    num_rot_dims = rotation_ndim(rotation_type)
    batch_shape = tuple(pose.shape[: -(num_rot_dims + 1)])
    bind_skeleton = xp.broadcast_to(bind_skeleton, (*batch_shape, *bind_skeleton.shape[-2:]))
    local_bind_translations = xp.broadcast_to(
        local_bind_translations,
        (*batch_shape, *local_bind_translations.shape[-2:]),
    )
    skeleton = _forward_skeleton_se3(
        bind_quats=bind_quats,
        local_bind_translations=local_bind_translations,
        kinematic_fronts=kinematic_fronts,
        pose=pose,
        rotation_type=rotation_type,
        xp=xp,
    )
    prepared: GarmentMeasurementsPreparedPose = {"skeleton_transforms": SE3.to_matrix(skeleton, xp=xp)}
    if not skip_vertices:
        skinning = SE3.multiply(skeleton, SE3.inverse(bind_skeleton, xp=xp), xp=xp)
        prepared["skinning_transforms"] = SE3.to_matrix(skinning, xp=xp)
    return prepared


def prepare_identity(
    *,
    xp: Any,
    mean_vertices: Float[Array, "V 3"],
    components: Float[Array, "V 3 C"],
    eigenvalues: Float[Array, "C"],
    bind_quats: Float[Array, "J 4"],
    mvc_weights: Float[Array, "V J"],
    kinematic_fronts: list[Front],
    shape: Float[Array, "*batch C"],
    skip_vertices: bool,
) -> GarmentMeasurementsIdentity:
    """Prepare shape-dependent surface and bind skeleton."""
    if shape.ndim < 1 or shape.shape[-1] != eigenvalues.shape[0]:
        raise ValueError(f"shape must have shape [..., {eigenvalues.shape[0]}]")
    scaled_shape = shape * xp.sqrt(eigenvalues)
    rest_vertices = mean_vertices + xp.einsum("...c,vdc->...vd", scaled_shape, components)
    joint_positions = xp.einsum("vj,...vd->...jd", mvc_weights, rest_vertices)
    bind_quats = xp.broadcast_to(bind_quats, (*rest_vertices.shape[:-2], *bind_quats.shape))
    bind_global_quats = _propagate_quats(bind_quats, kinematic_fronts, xp=xp)
    local_translations = _local_translations_from_positions(
        joint_positions,
        bind_global_quats,
        kinematic_fronts,
        xp=xp,
    )
    identity: GarmentMeasurementsIdentity = {
        "bind_skeleton": SE3.from_rt(bind_global_quats, joint_positions, xp=xp),
        "local_bind_translations": local_translations,
    }
    if not skip_vertices:
        identity["rest_vertices"] = rest_vertices
    return identity


def _forward_skeleton_se3(
    *,
    bind_quats: Float[Array, "J 4"],
    local_bind_translations: Float[Array, "*batch J 3"],
    kinematic_fronts: list[Front],
    pose: Float[Array, "*batch J N"] | Float[Array, "*batch J 3 3"],
    rotation_type: RotationType,
    xp: Any,
) -> Float[Array, "*batch J 7"]:
    batch_shape = local_bind_translations.shape[:-2]
    bind_quats = xp.broadcast_to(bind_quats, (*batch_shape, *bind_quats.shape))
    pose_quats = SO3.convert(pose, src=rotation_type, dst="quat", xp=xp)
    posed_quats = SO3.multiply(bind_quats, pose_quats, xp=xp)
    local_transforms = SE3.from_rt(posed_quats, local_bind_translations, xp=xp)
    return _propagate_se3(local_transforms, kinematic_fronts, xp=xp)


def _local_translations_from_positions(
    positions: Float[Array, "*batch J 3"],
    bind_global_quats: Float[Array, "*batch J 4"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 3"]:
    translations = xp.zeros_like(positions)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            front = positions[..., joints, :]
        else:
            offsets = positions[..., joints, :] - positions[..., parents, :]
            parent_inverse = SO3.inverse(bind_global_quats[..., parents, :], xp=xp)
            front = SO3.rotate_points(parent_inverse, offsets[..., None, :], xp=xp).squeeze(-2)
        translations = common.set(
            translations,
            (..., joints, slice(None)),
            front,
            copy=False,
            xp=xp,
        )
    return translations


def _propagate_quats(
    quaternions: Float[Array, "*batch J 4"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 4"]:
    global_quaternions = xp.zeros_like(quaternions)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            front = quaternions[..., joints, :]
        else:
            front = SO3.multiply(
                global_quaternions[..., parents, :],
                quaternions[..., joints, :],
                xp=xp,
            )
        global_quaternions = common.set(
            global_quaternions,
            (..., joints, slice(None)),
            front,
            copy=False,
            xp=xp,
        )
    return global_quaternions


def _propagate_se3(
    transforms: Float[Array, "*batch J 7"],
    kinematic_fronts: list[Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 7"]:
    global_transforms = xp.zeros_like(transforms)
    for joints, parents in kinematic_fronts:
        if parents[0] < 0:
            front = transforms[..., joints, :]
        else:
            front = SE3.multiply(
                global_transforms[..., parents, :],
                transforms[..., joints, :],
                xp=xp,
            )
        global_transforms = common.set(
            global_transforms,
            (..., joints, slice(None)),
            front,
            copy=False,
            xp=xp,
        )
    return global_transforms


__all__ = [
    "GarmentMeasurementsIdentity",
    "GarmentMeasurementsPreparedPose",
    "prepare_identity",
    "prepare_pose",
]
