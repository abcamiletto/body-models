"""Shared skeleton utilities."""

from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models.common import ops

Array = Any

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


def forward_kinematics(
    rotations: Float[Array, "*batch J 3 3"],
    translations: Float[Array, "*batch J 3"],
    fronts: list[Front],
    joint_indices: list[int] | None = None,
    *,
    xp: Any = None,
) -> Float[Array, "*batch J 4 4"]:
    """Compose local joint transforms into world-space transforms."""
    if xp is None:
        xp = ops.get_namespace(rotations)

    num_joints = rotations.shape[-3]
    batch_shape = rotations.shape[:-3]
    upper = xp.concat([rotations, translations[..., None]], axis=-1)
    bottom = ops.zeros_as(upper, shape=(*batch_shape, num_joints, 1, 4), xp=xp)
    bottom = ops.set(bottom, (..., 0, 3), 1.0, xp=xp)
    local_transforms = xp.concat([upper, bottom], axis=-2)

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

    if joint_indices is None:
        return xp.stack(world_transforms, axis=-3)
    return xp.stack([world_transforms[joint] for joint in joint_indices], axis=-3)


def compute_kinematic_fronts(parents: np.ndarray | list[int]) -> list[Front]:
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
        fronts.append((joints, joint_parents))
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
    indices = np.full((weights.shape[0], int(counts.max(initial=0))), -1, dtype=np.int64)
    values = np.zeros(indices.shape, dtype=weights.dtype)
    for vertex, row in enumerate(weights):
        active = np.flatnonzero(np.abs(row) > threshold)
        indices[vertex, : len(active)] = active
        values[vertex, : len(active)] = row[active]
    return indices, values
