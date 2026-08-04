"""Host-side preparation of pose-corrective model assets."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Int


def select_blocks(
    posedirs: Float[np.ndarray, "C V3"],
    joint_names: Sequence[str],
    pose_corrective_joints: Sequence[str] | None,
) -> tuple[Float[np.ndarray, "selected V3"], Int[np.ndarray, "selected"] | None, tuple[str, ...]]:
    """Select complete joint blocks, pruning zero blocks only by default."""
    if posedirs.shape[0] % 9:
        raise ValueError("pose corrective coefficient count must be divisible by 9")
    num_blocks = posedirs.shape[0] // 9
    if len(joint_names) != num_blocks + 1:
        raise ValueError("pose corrective blocks must align with the root-excluded joint list")
    if isinstance(pose_corrective_joints, str):
        raise TypeError("pose_corrective_joints must be a sequence of joint names, not a string")

    available_names = tuple(joint_names[1:])
    requested_names = available_names if pose_corrective_joints is None else tuple(pose_corrective_joints)
    if any(not isinstance(name, str) for name in requested_names):
        raise TypeError("pose_corrective_joints must contain only joint names")
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("pose_corrective_joints must not contain duplicate names")
    unknown = sorted(set(requested_names) - set(available_names))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown or root pose-corrective joints: {names}")

    requested = set(requested_names)
    remove_zero_blocks = pose_corrective_joints is None
    blocks = posedirs.reshape(num_blocks, 9, posedirs.shape[1])
    selected_blocks = np.asarray(
        [
            index
            for index, (name, block) in enumerate(zip(available_names, blocks, strict=True))
            if name in requested and (not remove_zero_blocks or np.any(block))
        ],
        dtype=np.int64,
    )
    coefficient_indices = None
    if len(selected_blocks) != num_blocks:
        coefficient_indices = (selected_blocks[:, None] * 9 + np.arange(9)).reshape(-1)
    selected_posedirs = blocks[selected_blocks].reshape(-1, posedirs.shape[1])
    selected_names = tuple(available_names[index] for index in selected_blocks)
    return selected_posedirs, coefficient_indices, selected_names


__all__ = ["select_blocks"]
