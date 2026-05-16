"""Rigid body model visualization helpers for ``viser``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from body_models.base import BodyModel
from body_models.extras.viser_plugin.skeleton import ViserSkeletonHandle, add_skeleton

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
        params: dict[str, Float[np.ndarray, "..."]],
        root_frame: viser.FrameHandle,
        links: list[viser.MeshHandle],
        skeleton: ViserSkeletonHandle | None = None,
    ) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self._params = params
        self.root_frame = root_frame
        self.links = links
        self.skeleton = skeleton

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
    def shape(self) -> Float[np.ndarray, "..."]:
        assert "shape" in self._params, f"{self.model_name} does not support 'shape'."
        return self._params["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "shape" in self._params, f"{self.model_name} does not support 'shape'."
        self._params["shape"] = np.asarray(value)
        self._apply_params()

    @property
    def body_pose(self) -> Float[np.ndarray, "..."]:
        assert "body_pose" in self._params, f"{self.model_name} does not support 'body_pose'."
        return self._params["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "body_pose" in self._params, f"{self.model_name} does not support 'body_pose'."
        self._params["body_pose"] = np.asarray(value)
        self._apply_params()

    @property
    def hand_pose(self) -> Float[np.ndarray, "..."]:
        assert "hand_pose" in self._params, f"{self.model_name} does not support 'hand_pose'."
        return self._params["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "hand_pose" in self._params, f"{self.model_name} does not support 'hand_pose'."
        self._params["hand_pose"] = np.asarray(value)
        self._apply_params()

    @property
    def head_pose(self) -> Float[np.ndarray, "..."]:
        assert "head_pose" in self._params, f"{self.model_name} does not support 'head_pose'."
        return self._params["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "head_pose" in self._params, f"{self.model_name} does not support 'head_pose'."
        self._params["head_pose"] = np.asarray(value)
        self._apply_params()

    @property
    def expression(self) -> Float[np.ndarray, "..."]:
        assert "expression" in self._params, f"{self.model_name} does not support 'expression'."
        return self._params["expression"]

    @expression.setter
    def expression(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "expression" in self._params, f"{self.model_name} does not support 'expression'."
        self._params["expression"] = np.asarray(value)
        self._apply_params()

    @property
    def global_rotation(self) -> Float[np.ndarray, "..."]:
        assert "global_rotation" in self._params, f"{self.model_name} does not support 'global_rotation'."
        return self._params["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_rotation" in self._params, f"{self.model_name} does not support 'global_rotation'."
        self._params["global_rotation"] = np.asarray(value)
        self._apply_params()

    @property
    def global_translation(self) -> Float[np.ndarray, "..."]:
        assert "global_translation" in self._params, f"{self.model_name} does not support 'global_translation'."
        return self._params["global_translation"]

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_translation" in self._params, f"{self.model_name} does not support 'global_translation'."
        self._params["global_translation"] = np.asarray(value)
        self._apply_params()

    def _apply_params(self) -> None:
        transforms = np.asarray(self.model.forward_links(**self._params))  # type: ignore[attr-defined]
        rotations = transforms[:, :3, :3]
        wxyzs = SO3.conversions.from_rotmat_to_quat(rotations, convention="wxyz", xp=np)
        positions = transforms[:, :3, 3]
        for handle, wxyz, position in zip(self.links, wxyzs, positions):
            handle.wxyz = wxyz
            handle.position = position
        if self.skeleton is not None:
            self.skeleton.skeleton = self.model.forward_skeleton(**self._params)

    def remove(self) -> None:
        for handle in self.links:
            handle.remove()
        if self.skeleton is not None:
            self.skeleton.remove()
        self.root_frame.remove()


def add_rigid_body_model(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
    show_skeleton: bool = False,
    skeleton_color: tuple[float, float, float] = (120, 180, 255),
) -> ViserRigidBodyModelHandle:
    """Add a rigid articulated body model to a ``viser`` scene."""
    if not model.is_rigid_body:
        model_name = model.__class__.__name__
        raise ValueError(f"add_rigid_body_model() only supports rigid models, got {model_name}.")

    params = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)
    skeleton = None
    if show_skeleton:
        skeleton_path = f"{name}/skeleton"
        skeleton = add_skeleton(scene, skeleton_path, model, color=skeleton_color)

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
    handle = ViserRigidBodyModelHandle(model, params, root, links, skeleton)
    handle._apply_params()
    return handle
