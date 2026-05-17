"""Rigid body model visualization helpers for ``viser``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from body_models.base import BodyModel

if TYPE_CHECKING:
    import viser


__all__ = [
    "ViserRigidBodyModelHandle",
    "add_rigid_body_model",
]


class ViserRigidBodyModelHandle:
    """Rigid articulated body model rendered as one static mesh per link."""

    def __init__(
        self,
        model: BodyModel,
        pose: dict[str, Float[np.ndarray, "..."]],
        root_frame: viser.FrameHandle,
        links: list[viser.MeshHandle],
    ) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self.pose = pose
        self.root_frame = root_frame
        self.links = links

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

    @property
    def body_pose(self) -> Float[np.ndarray, "..."]:
        assert "body_pose" in self.pose, f"{self.model_name} does not support 'body_pose'."
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "body_pose" in self.pose, f"{self.model_name} does not support 'body_pose'."
        self.pose["body_pose"] = np.asarray(value)
        self._apply_pose()

    @property
    def hand_pose(self) -> Float[np.ndarray, "..."]:
        assert "hand_pose" in self.pose, f"{self.model_name} does not support 'hand_pose'."
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "hand_pose" in self.pose, f"{self.model_name} does not support 'hand_pose'."
        self.pose["hand_pose"] = np.asarray(value)
        self._apply_pose()

    @property
    def global_rotation(self) -> Float[np.ndarray, "..."]:
        assert "global_rotation" in self.pose, f"{self.model_name} does not support 'global_rotation'."
        return self.pose["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_rotation" in self.pose, f"{self.model_name} does not support 'global_rotation'."
        self.pose["global_rotation"] = np.asarray(value)
        self._apply_pose()

    @property
    def global_translation(self) -> Float[np.ndarray, "..."]:
        assert "global_translation" in self.pose, f"{self.model_name} does not support 'global_translation'."
        return self.pose["global_translation"]

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_translation" in self.pose, f"{self.model_name} does not support 'global_translation'."
        self.pose["global_translation"] = np.asarray(value)
        self._apply_pose()

    def set_pose(self, **forward_kwargs: Float[np.ndarray, "..."] | np.ndarray) -> None:
        changed = False
        for name, value in forward_kwargs.items():
            assert name in self.pose, f"{self.model_name} does not support {name!r}."
            value = np.asarray(value)
            if np.array_equal(self.pose[name], value):
                continue
            self.pose[name] = value.copy()
            changed = True
        if not changed:
            return
        self._apply_pose()

    def _apply_pose(self) -> None:
        transforms = np.asarray(self.model.forward_links(**self.pose))  # type: ignore[attr-defined]
        rotations = transforms[:, :3, :3]
        wxyzs = SO3.conversions.from_rotmat_to_quat(rotations, convention="wxyz", xp=np)
        positions = transforms[:, :3, 3]
        for handle, wxyz, position in zip(self.links, wxyzs, positions):
            handle.wxyz = wxyz
            handle.position = position

    def remove(self) -> None:
        for handle in self.links:
            handle.remove()
        self.root_frame.remove()


def add_rigid_body_model(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
) -> ViserRigidBodyModelHandle:
    """Add a rigid articulated body model to a ``viser`` scene."""
    if not model.is_rigid_body:
        model_name = model.__class__.__name__
        raise ValueError(f"add_rigid_body_model() only supports rigid models, got {model_name}.")

    pose = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)

    links = []
    for index, link_name in enumerate(model.link_names):  # type: ignore[attr-defined]
        mesh = model.link_mesh(link_name)  # type: ignore[attr-defined]
        link_path = f"{name}/links/{index:03d}"
        links.append(
            scene.add_mesh_simple(
                link_path,
                vertices=np.asarray(mesh["vertices"], dtype=np.float32),
                faces=np.asarray(mesh["faces"]),
                color=color,
            )
        )
    handle = ViserRigidBodyModelHandle(model, pose, root, links)
    handle._apply_pose()
    return handle
