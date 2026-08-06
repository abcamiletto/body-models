"""Shared skeleton utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models._common import ops

Array = Any

Front = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class KinematicTree:
    """Immutable parent tree and its depth-parallel traversal."""

    parents: tuple[int, ...]
    fronts: tuple[Front, ...]

    @classmethod
    def from_parents(cls, parents: Int[np.ndarray, "J"] | Sequence[int]) -> KinematicTree:
        parent_tuple = tuple(int(parent) for parent in parents)
        return cls(parent_tuple, tuple(compute_kinematic_fronts(parent_tuple)))


def affine_transforms(
    linear: Float[Array, "*batch 3 3"],
    translation: Float[Array, "*batch 3"] | None = None,
    *,
    xp: Any,
) -> Float[Array, "*batch 4 4"]:
    """Assemble homogeneous transforms from linear maps and translations."""
    if translation is None:
        translation = ops.zeros_as(linear, shape=(*linear.shape[:-2], 3), xp=xp)

    batch_shape = np.broadcast_shapes(linear.shape[:-2], translation.shape[:-1])
    linear = xp.broadcast_to(linear, (*batch_shape, 3, 3))
    translation = xp.broadcast_to(translation, (*batch_shape, 3))
    upper = xp.concat([linear, translation[..., None]], axis=-1)
    bottom = ops.zeros_as(upper, shape=(*batch_shape, 1, 4), xp=xp)
    bottom = ops.at_set(bottom, (..., 0, 3), 1.0, xp=xp)
    return xp.concat([upper, bottom], axis=-2)


def invert_rigid_transforms(
    transforms: Float[Array, "*batch 4 4"],
    *,
    xp: Any,
) -> Float[Array, "*batch 4 4"]:
    """Invert homogeneous transforms whose linear part is a rotation."""
    rotations = transforms[..., :3, :3]
    translations = transforms[..., :3, 3]
    inverse_rotations = rotations.mT
    inverse_translations = -xp.squeeze(inverse_rotations @ translations[..., None], axis=-1)
    return affine_transforms(inverse_rotations, inverse_translations, xp=xp)


def local_joint_offsets(
    joints: Float[Array, "*batch J 3"],
    parents: Sequence[int],
    *,
    xp: Any,
) -> Float[Array, "*batch J 3"]:
    """Convert world-space rest joints to parent-relative translations."""
    if len(parents) != joints.shape[-2]:
        raise ValueError("parents must contain one entry per joint")

    roots = [joint for joint, parent in enumerate(parents) if parent < 0 or parent == joint]
    parent_indices = [joint if joint in roots else int(parent) for joint, parent in enumerate(parents)]
    offsets = joints - joints[..., parent_indices, :]
    if roots:
        offsets = ops.at_set(offsets, (..., roots, slice(None)), joints[..., roots, :], xp=xp)
    return offsets


def compose_kinematic_fronts(
    local_transforms: Float[Array, "*batch J 4 4"],
    fronts: Sequence[Front],
    *,
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Compose local transforms into world-space transforms."""
    num_joints = local_transforms.shape[-3]

    world_transforms: list[Float[Array, "*batch 4 4"] | None] = [None] * num_joints
    for joints, parents in fronts:
        if parents[0] < 0:
            for joint in joints:
                world_transforms[joint] = local_transforms[..., joint, :, :]
            continue

        parent_transforms = xp.stack([world_transforms[parent] for parent in parents], axis=-3)
        front_transforms = parent_transforms @ local_transforms[..., joints, :, :]
        for index, joint in enumerate(joints):
            world_transforms[joint] = front_transforms[..., index, :, :]

    return xp.stack(world_transforms, axis=-3)


def compute_kinematic_fronts(parents: Int[np.ndarray, "J"] | Sequence[int]) -> list[Front]:
    """Group joints by depth for parallel forward kinematics.

    Roots are joints with ``parent < 0`` or ``parent == joint``; they are
    reported with parent ``-1``. Raises on cyclic parent chains.
    """
    parents_list = [int(p) for p in parents]
    processed: set[int] = set()
    fronts: list[Front] = []
    while len(processed) < len(parents_list):
        joints: list[int] = []
        joint_parents: list[int] = []
        for j, parent in enumerate(parents_list):
            if j in processed:
                continue
            if parent < 0 or parent == j or parent in processed:
                joints.append(j)
                joint_parents.append(-1 if parent < 0 or parent == j else parent)
        if not joints:
            raise ValueError(f"Invalid parent chain: {parents_list}")
        fronts.append((tuple(joints), tuple(joint_parents)))
        processed.update(joints)
    return fronts


def compute_sparse_skin_weights(
    weights: Float[np.ndarray, "V J"],
    threshold: float = 1e-8,
) -> tuple[Int[np.ndarray, "V K"], Float[np.ndarray, "V K"]]:
    """Compact dense per-vertex joint weights into (indices, weights) slots.

    K is the max active joints of any vertex; unused slots have index -1 and
    weight 0.
    """
    counts = (np.abs(weights) > threshold).sum(axis=1)
    indices = np.full((weights.shape[0], int(counts.max(initial=0))), -1, dtype=np.int32)
    values = np.zeros(indices.shape, dtype=weights.dtype)
    for vertex, row in enumerate(weights):
        active = np.flatnonzero(np.abs(row) > threshold)
        indices[vertex, : len(active)] = active
        values[vertex, : len(active)] = row[active]
    return indices, values
