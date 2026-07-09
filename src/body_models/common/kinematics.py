"""Shared skeleton utilities: kinematic fronts and sparse skin weights."""

import numpy as np
from jaxtyping import Float, Int

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).


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

    K is the max active joints of any vertex; unused slots have index 0 and
    weight 0.
    """
    counts = (np.abs(weights) > threshold).sum(axis=1)
    indices = np.zeros((weights.shape[0], int(counts.max(initial=0))), dtype=np.int64)
    values = np.zeros(indices.shape, dtype=weights.dtype)
    for vertex, row in enumerate(weights):
        active = np.flatnonzero(np.abs(row) > threshold)
        indices[vertex, : len(active)] = active
        values[vertex, : len(active)] = row[active]
    return indices, values
