"""Pose packing layouts and their canonical skeleton joints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PoseLayout:
    """Canonical runs of pose controls and the local joints they drive."""

    runs: tuple[tuple[str | None, int], ...]
    control_joints: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def per_joint(cls, *runs: tuple[str | None, int]) -> PoseLayout:
        count = sum(size for _, size in runs)
        return cls(runs, tuple((index,) for index in range(count)))

    def with_control_joints(self, control_joints: tuple[tuple[int, ...], ...]) -> PoseLayout:
        return PoseLayout(self.runs, control_joints)

    @property
    def num_controls(self) -> int:
        return sum(count for _, count in self.runs)

    @property
    def joint_indices(self) -> Mapping[str, tuple[int, ...]]:
        if len(self.control_joints) != self.num_controls:
            raise ValueError("Pose layout has no joint mapping")
        grouped: dict[str, list[int]] = {}
        offset = 0
        for name, count in self.runs:
            if name is not None:
                joints = grouped.setdefault(name, [])
                for control_joints in self.control_joints[offset : offset + count]:
                    joints.extend(control_joints)
            offset += count
        return {name: tuple(dict.fromkeys(joints)) for name, joints in grouped.items()}

    def pack(self, xp: Any, values: Mapping[str | None, Any], *, axis: int) -> Any:
        offsets: dict[str | None, int] = {}
        parts = []
        for name, count in self.runs:
            start = offsets.get(name, 0)
            stop = start + count
            index = [slice(None)] * values[name].ndim
            index[axis] = slice(start, stop)
            parts.append(values[name][tuple(index)])
            offsets[name] = stop
        return xp.concat(parts, axis=axis)

    def unpack(self, xp: Any, pose: Any, *, axis: int) -> dict[str | None, Any]:
        parts: dict[str | None, list[Any]] = {}
        offset = 0
        for name, count in self.runs:
            index = [slice(None)] * pose.ndim
            index[axis] = slice(offset, offset + count)
            parts.setdefault(name, []).append(pose[tuple(index)])
            offset += count
        return {name: values[0] if len(values) == 1 else xp.concat(values, axis=axis) for name, values in parts.items()}


__all__ = ["PoseLayout"]
