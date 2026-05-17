"""Skeleton visualization helpers for ``viser``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float, Int

if TYPE_CHECKING:
    import viser


__all__ = [
    "ViserSkeletonHandle",
    "add_skeleton",
]


class ViserSkeletonHandle:
    """Skeleton rendered as joint markers and cylinder bones."""

    def __init__(
        self,
        root_frame: viser.FrameHandle,
        parent_tree: list[tuple[int, int]],
        bones: list[viser.CylinderHandle],
        joint_positions: Float[np.ndarray, "N 3"],
        joints: list[viser.IcosphereHandle],
        joint_names: tuple[str, ...],
    ) -> None:
        self.root_frame = root_frame
        self.parent_tree = parent_tree
        self.bones = bones
        self._joint_positions = joint_positions
        self.joints = joints
        self.joint_names = joint_names

    @property
    def name(self) -> str:
        return self.root_frame.name

    @property
    def wxyz(self) -> Float[np.ndarray, "4"]:
        return self.root_frame.wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (4,)
        self.root_frame.wxyz = value

    @property
    def position(self) -> Float[np.ndarray, "3"]:
        return self.root_frame.position

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (3,)
        self.root_frame.position = value

    @property
    def visible(self) -> bool:
        return self.root_frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.root_frame.visible = value
        for handle in self.joints:
            handle.visible = value
        for handle in self.bones:
            handle.visible = value

    @property
    def joint_positions(self) -> Float[np.ndarray, "N 3"]:
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, value: Float[np.ndarray, "N 3"]) -> None:
        self._joint_positions = np.asarray(value)
        for handle, position in zip(self.joints, self._joint_positions):
            handle.position = position
        for handle, (parent, child) in zip(self.bones, self.parent_tree):
            position, wxyz, height = _cylinder_between(self._joint_positions[parent], self._joint_positions[child])
            handle.position = position
            handle.wxyz = wxyz
            handle.height = height

    def remove(self) -> None:
        for handle in self.joints:
            handle.remove()
        for handle in self.bones:
            handle.remove()
        self.root_frame.remove()


def add_skeleton(
    scene: viser.SceneApi,
    name: str,
    joint_positions: Float[np.ndarray, "N 3"],
    parents: Int[np.ndarray, "N"] | list[int] | tuple[int, ...],
    *,
    joint_names: tuple[str, ...] | None = None,
    color: tuple[float, float, float] = (120, 180, 255),
    joint_color: tuple[float, float, float] = (255, 255, 255),
    bone_radius: float = 0.006,
    joint_radius: float = 0.015,
) -> ViserSkeletonHandle:
    """Add a clickable skeleton to a ``viser`` scene."""
    joint_positions = np.asarray(joint_positions)
    parents = [int(parent) for parent in parents]
    joint_names = joint_names or tuple(f"joint_{index}" for index in range(len(joint_positions)))
    parent_tree = [(parent, index) for index, parent in enumerate(parents) if parent >= 0]
    root = scene.add_frame(name, show_axes=False)

    bones = []
    for index, (parent, child) in enumerate(parent_tree):
        position, wxyz, height = _cylinder_between(joint_positions[parent], joint_positions[child])
        bones.append(
            scene.add_cylinder(
                f"{name}/bones/{index:03d}",
                radius=bone_radius,
                height=height,
                color=color,
                position=position,
                wxyz=wxyz,
            )
        )

    joints = []
    for index, position in enumerate(joint_positions):
        joints.append(
            scene.add_icosphere(
                f"{name}/joints/{index:03d}",
                radius=joint_radius,
                color=joint_color,
                position=position,
            )
        )
    for index, joint in enumerate(joints):

        @joint.on_click
        def _(event, joint_index=index) -> None:
            event.client.add_notification(
                title=f"Joint: {joint_names[joint_index]}",
                body="",
                auto_close_seconds=3,
            )

    return ViserSkeletonHandle(root, parent_tree, bones, joint_positions, joints, joint_names)


def _cylinder_between(
    p0: Float[np.ndarray, "3"],
    p1: Float[np.ndarray, "3"],
) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "4"], float]:
    diff = p1 - p0
    height = float(np.linalg.norm(diff))
    direction = diff / height
    dot = float(direction[2])

    if dot > 0.9999:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif dot < -0.9999:
        wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
        axis /= np.linalg.norm(axis)
        half = np.arccos(dot) / 2.0
        wxyz = np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float32)

    return (p0 + p1) / 2.0, wxyz, height
