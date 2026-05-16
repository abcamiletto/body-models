"""Optional ``viser`` helpers for body model visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from body_models.base import BodyModel

if TYPE_CHECKING:
    import viser


POSE_PARAMS = {
    "body_pose",
    "hand_pose",
    "head_pose",
    "pelvis_rotation",
    "head_rotation",
    "wrist_rotation",
    "global_rotation",
    "global_translation",
}

__all__ = [
    "ViserBodyModelHandle",
    "ViserRigidBodyModelHandle",
    "ViserSkeletonHandle",
    "add_body_model",
    "add_rigid_body_model",
    "add_skeleton",
]


class ViserSkeletonHandle:
    """Skeleton rendered as joint markers and cylinder bones."""

    def __init__(
        self,
        root_frame: viser.FrameHandle,
        parent_tree: list[tuple[int, int]],
        bones: list[viser.CylinderHandle],
        skeleton: Float[np.ndarray, "N 4 4"],
        joints: list[viser.IcosphereHandle],
    ) -> None:
        self.root_frame = root_frame
        self.parent_tree = parent_tree
        self.bones = bones
        self._skeleton = skeleton
        self.joints = joints

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
    def skeleton(self) -> Float[np.ndarray, "N 4 4"]:
        return self._skeleton

    @skeleton.setter
    def skeleton(self, value: Float[np.ndarray, "N 4 4"]) -> None:
        self._skeleton = value
        positions = _skeleton_positions(self._skeleton)
        for handle, position in zip(self.joints, positions):
            position = np.asarray(position, dtype=np.float32)
            assert position.shape == (3,)
            handle.position = position
        for handle, (parent, child) in zip(self.bones, self.parent_tree):
            position, wxyz, height = _cylinder_between(positions[parent], positions[child])
            handle.position = position
            handle.wxyz = wxyz
            handle.height = height

    def remove(self) -> None:
        for handle in self.joints:
            handle.remove()
        for handle in self.bones:
            handle.remove()
        self.root_frame.remove()


class ViserBodyModelHandle:
    """Non-rigid body model rendered as a skinned mesh or simple mesh."""

    def __init__(
        self,
        model: BodyModel,
        params: dict[str, object],
        root_frame: viser.FrameHandle,
        mesh: viser.MeshHandle | viser.MeshSkinnedHandle,
        *,
        skinned: bool,
        skeleton: ViserSkeletonHandle | None = None,
    ) -> None:
        self.model = model
        self._params = params
        self._rest_params = dict(params)
        self.root_frame = root_frame
        self.mesh = mesh
        self.skinned = skinned
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
    def shape(self) -> object:
        if "shape" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'shape'.")
        return self._params["shape"]

    @shape.setter
    def shape(self, value: object) -> None:
        if "shape" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'shape'.")
        self._params["shape"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def body_pose(self) -> object:
        if "body_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'body_pose'.")
        return self._params["body_pose"]

    @body_pose.setter
    def body_pose(self, value: object) -> None:
        if "body_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'body_pose'.")
        self._params["body_pose"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def hand_pose(self) -> object:
        if "hand_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'hand_pose'.")
        return self._params["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: object) -> None:
        if "hand_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'hand_pose'.")
        self._params["hand_pose"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def head_pose(self) -> object:
        if "head_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_pose'.")
        return self._params["head_pose"]

    @head_pose.setter
    def head_pose(self, value: object) -> None:
        if "head_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_pose'.")
        self._params["head_pose"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def expression(self) -> object:
        if "expression" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'expression'.")
        return self._params["expression"]

    @expression.setter
    def expression(self, value: object) -> None:
        if "expression" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'expression'.")
        self._params["expression"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def identity(self) -> object:
        if "identity" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'identity'.")
        return self._params["identity"]

    @identity.setter
    def identity(self, value: object) -> None:
        if "identity" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'identity'.")
        self._params["identity"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def pelvis_rotation(self) -> object:
        if "pelvis_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'pelvis_rotation'.")
        return self._params["pelvis_rotation"]

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: object) -> None:
        if "pelvis_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'pelvis_rotation'.")
        self._params["pelvis_rotation"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def head_rotation(self) -> object:
        if "head_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_rotation'.")
        return self._params["head_rotation"]

    @head_rotation.setter
    def head_rotation(self, value: object) -> None:
        if "head_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_rotation'.")
        self._params["head_rotation"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def wrist_rotation(self) -> object:
        if "wrist_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'wrist_rotation'.")
        return self._params["wrist_rotation"]

    @wrist_rotation.setter
    def wrist_rotation(self, value: object) -> None:
        if "wrist_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'wrist_rotation'.")
        self._params["wrist_rotation"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def global_rotation(self) -> object:
        if "global_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_rotation'.")
        return self._params["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: object) -> None:
        if "global_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_rotation'.")
        self._params["global_rotation"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def global_translation(self) -> object:
        if "global_translation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_translation'.")
        return self._params["global_translation"]

    @global_translation.setter
    def global_translation(self, value: object) -> None:
        if "global_translation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_translation'.")
        self._params["global_translation"] = value
        self._apply_params(rebuild_mesh=False)

    @property
    def gender(self) -> object:
        if "gender" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'gender'.")
        return self._params["gender"]

    @gender.setter
    def gender(self, value: object) -> None:
        if "gender" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'gender'.")
        self._params["gender"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def age(self) -> object:
        if "age" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'age'.")
        return self._params["age"]

    @age.setter
    def age(self, value: object) -> None:
        if "age" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'age'.")
        self._params["age"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def muscle(self) -> object:
        if "muscle" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'muscle'.")
        return self._params["muscle"]

    @muscle.setter
    def muscle(self, value: object) -> None:
        if "muscle" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'muscle'.")
        self._params["muscle"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def weight(self) -> object:
        if "weight" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'weight'.")
        return self._params["weight"]

    @weight.setter
    def weight(self, value: object) -> None:
        if "weight" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'weight'.")
        self._params["weight"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def height(self) -> object:
        if "height" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'height'.")
        return self._params["height"]

    @height.setter
    def height(self, value: object) -> None:
        if "height" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'height'.")
        self._params["height"] = value
        self._apply_params(rebuild_mesh=True)

    @property
    def proportions(self) -> object:
        if "proportions" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'proportions'.")
        return self._params["proportions"]

    @proportions.setter
    def proportions(self, value: object) -> None:
        if "proportions" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'proportions'.")
        self._params["proportions"] = value
        self._apply_params(rebuild_mesh=True)

    def _apply_params(self, *, rebuild_mesh: bool = False) -> None:
        if self.skinned:
            if rebuild_mesh:
                bind_params = dict(self._params)
                for name in POSE_PARAMS:
                    if name in bind_params and name in self._rest_params:
                        bind_params[name] = self._rest_params[name]
                vertices = np.asarray(self.model.forward_vertices(**bind_params))
                self.mesh.vertices = _unbatched_vertices(vertices, "forward_vertices")
            bones = self.model.to_viser_bones(**self._params)
            for bone, wxyz, position in zip(
                self.mesh.bones,  # type: ignore[possibly-missing-attribute]
                bones["bone_wxyzs"],
                bones["bone_positions"],
            ):
                bone.wxyz = wxyz
                position = np.asarray(position, dtype=np.float32)
                assert position.shape == (3,)
                bone.position = position
        else:
            vertices = np.asarray(self.model.forward_vertices(**self._params))
            self.mesh.vertices = _unbatched_vertices(vertices, "forward_vertices")

        if self.skeleton is not None:
            self.skeleton.skeleton = self.model.forward_skeleton(**self._params)

    def remove(self) -> None:
        self.mesh.remove()
        if self.skeleton is not None:
            self.skeleton.remove()
        self.root_frame.remove()


class ViserRigidBodyModelHandle:
    """Rigid articulated body model rendered as one static mesh per link."""

    def __init__(
        self,
        model: BodyModel,
        params: dict[str, object],
        root_frame: viser.FrameHandle,
        links: list[viser.MeshHandle],
        skeleton: ViserSkeletonHandle | None = None,
    ) -> None:
        self.model = model
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
    def shape(self) -> object:
        if "shape" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'shape'.")
        return self._params["shape"]

    @shape.setter
    def shape(self, value: object) -> None:
        if "shape" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'shape'.")
        self._params["shape"] = value
        self._apply_params()

    @property
    def body_pose(self) -> object:
        if "body_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'body_pose'.")
        return self._params["body_pose"]

    @body_pose.setter
    def body_pose(self, value: object) -> None:
        if "body_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'body_pose'.")
        self._params["body_pose"] = value
        self._apply_params()

    @property
    def hand_pose(self) -> object:
        if "hand_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'hand_pose'.")
        return self._params["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: object) -> None:
        if "hand_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'hand_pose'.")
        self._params["hand_pose"] = value
        self._apply_params()

    @property
    def head_pose(self) -> object:
        if "head_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_pose'.")
        return self._params["head_pose"]

    @head_pose.setter
    def head_pose(self, value: object) -> None:
        if "head_pose" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_pose'.")
        self._params["head_pose"] = value
        self._apply_params()

    @property
    def expression(self) -> object:
        if "expression" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'expression'.")
        return self._params["expression"]

    @expression.setter
    def expression(self, value: object) -> None:
        if "expression" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'expression'.")
        self._params["expression"] = value
        self._apply_params()

    @property
    def identity(self) -> object:
        if "identity" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'identity'.")
        return self._params["identity"]

    @identity.setter
    def identity(self, value: object) -> None:
        if "identity" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'identity'.")
        self._params["identity"] = value
        self._apply_params()

    @property
    def pelvis_rotation(self) -> object:
        if "pelvis_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'pelvis_rotation'.")
        return self._params["pelvis_rotation"]

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: object) -> None:
        if "pelvis_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'pelvis_rotation'.")
        self._params["pelvis_rotation"] = value
        self._apply_params()

    @property
    def head_rotation(self) -> object:
        if "head_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_rotation'.")
        return self._params["head_rotation"]

    @head_rotation.setter
    def head_rotation(self, value: object) -> None:
        if "head_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'head_rotation'.")
        self._params["head_rotation"] = value
        self._apply_params()

    @property
    def wrist_rotation(self) -> object:
        if "wrist_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'wrist_rotation'.")
        return self._params["wrist_rotation"]

    @wrist_rotation.setter
    def wrist_rotation(self, value: object) -> None:
        if "wrist_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'wrist_rotation'.")
        self._params["wrist_rotation"] = value
        self._apply_params()

    @property
    def global_rotation(self) -> object:
        if "global_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_rotation'.")
        return self._params["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: object) -> None:
        if "global_rotation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_rotation'.")
        self._params["global_rotation"] = value
        self._apply_params()

    @property
    def global_translation(self) -> object:
        if "global_translation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_translation'.")
        return self._params["global_translation"]

    @global_translation.setter
    def global_translation(self, value: object) -> None:
        if "global_translation" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'global_translation'.")
        self._params["global_translation"] = value
        self._apply_params()

    @property
    def gender(self) -> object:
        if "gender" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'gender'.")
        return self._params["gender"]

    @gender.setter
    def gender(self, value: object) -> None:
        if "gender" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'gender'.")
        self._params["gender"] = value
        self._apply_params()

    @property
    def age(self) -> object:
        if "age" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'age'.")
        return self._params["age"]

    @age.setter
    def age(self, value: object) -> None:
        if "age" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'age'.")
        self._params["age"] = value
        self._apply_params()

    @property
    def muscle(self) -> object:
        if "muscle" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'muscle'.")
        return self._params["muscle"]

    @muscle.setter
    def muscle(self, value: object) -> None:
        if "muscle" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'muscle'.")
        self._params["muscle"] = value
        self._apply_params()

    @property
    def weight(self) -> object:
        if "weight" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'weight'.")
        return self._params["weight"]

    @weight.setter
    def weight(self, value: object) -> None:
        if "weight" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'weight'.")
        self._params["weight"] = value
        self._apply_params()

    @property
    def height(self) -> object:
        if "height" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'height'.")
        return self._params["height"]

    @height.setter
    def height(self, value: object) -> None:
        if "height" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'height'.")
        self._params["height"] = value
        self._apply_params()

    @property
    def proportions(self) -> object:
        if "proportions" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'proportions'.")
        return self._params["proportions"]

    @proportions.setter
    def proportions(self, value: object) -> None:
        if "proportions" not in self._params:
            raise AttributeError(f"{self.model.__class__.__name__} does not support 'proportions'.")
        self._params["proportions"] = value
        self._apply_params()

    def _apply_params(self, *, rebuild_mesh: bool = False) -> None:
        _update_rigid_links(self.model, self.links, **self._params)
        if self.skeleton is not None:
            self.skeleton.skeleton = self.model.forward_skeleton(**self._params)

    def remove(self) -> None:
        for handle in self.links:
            handle.remove()
        if self.skeleton is not None:
            self.skeleton.remove()
        self.root_frame.remove()


def add_skeleton(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (120, 180, 255),
    joint_color: tuple[float, float, float] | None = (255, 255, 255),
    bone_radius: float = 0.006,
    joint_radius: float = 0.015,
) -> ViserSkeletonHandle:
    """Add a model skeleton to a ``viser`` scene."""
    params = model.get_rest_pose()
    skeleton = model.forward_skeleton(**params)
    positions = _skeleton_positions(skeleton)
    parents = _parents(model)
    parent_tree = [(parent, index) for index, parent in enumerate(parents) if parent >= 0]
    root = scene.add_frame(name, show_axes=False)

    bones = []
    for index, (parent, child) in enumerate(parent_tree):
        position, wxyz, height = _cylinder_between(positions[parent], positions[child])
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
    if joint_color is not None:
        for index, position in enumerate(positions):
            position = np.asarray(position, dtype=np.float32)
            assert position.shape == (3,)
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
                title=f"Joint: {model.joint_names[joint_index]}",
                body="",
                auto_close_seconds=3,
            )

    return ViserSkeletonHandle(root, parent_tree, bones, np.asarray(skeleton), joints)


def add_body_model(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
    show_skeleton: bool = False,
    skeleton_color: tuple[float, float, float] = (120, 180, 255),
) -> ViserBodyModelHandle:
    """Add a non-rigid body model to a ``viser`` scene."""
    if model.is_rigid_body:
        raise ValueError("add_body_model() only supports non-rigid models; use add_rigid_body_model() instead.")

    params = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)
    skeleton = add_skeleton(scene, f"{name}/skeleton", model, color=skeleton_color) if show_skeleton else None

    if _supports_skinned_mesh(model):
        mesh = model.to_viser_skinned_mesh(**params)
        mesh_handle = scene.add_mesh_skinned(
            f"{name}/mesh",
            vertices=mesh["vertices"],
            faces=mesh["faces"],
            bone_wxyzs=mesh["bone_wxyzs"],
            bone_positions=mesh["bone_positions"],
            skin_weights=mesh["skin_weights"],
            color=color,
        )
        return ViserBodyModelHandle(model, params, root, mesh_handle, skinned=True, skeleton=skeleton)

    vertices = _unbatched_vertices(np.asarray(model.forward_vertices(**params)), "forward_vertices")
    mesh = scene.add_mesh_simple(f"{name}/mesh", vertices=vertices, faces=np.asarray(model.faces), color=color)
    return ViserBodyModelHandle(model, params, root, mesh, skinned=False, skeleton=skeleton)


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
        raise ValueError("add_rigid_body_model() only supports rigid models; use add_body_model() instead.")

    params = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)
    skeleton = add_skeleton(scene, f"{name}/skeleton", model, color=skeleton_color) if show_skeleton else None
    handle = ViserRigidBodyModelHandle(model, params, root, _add_rigid_links(scene, name, model, color), skeleton)
    handle._apply_params()
    return handle


def _supports_skinned_mesh(model: BodyModel) -> bool:
    if model.is_rigid_body:
        return False
    try:
        np.asarray(model.skin_weights)
    except (AttributeError, NotImplementedError):
        return False
    return True


def _add_rigid_links(
    scene: viser.SceneApi, name: str, model: BodyModel, color: tuple[float, float, float]
) -> list[viser.MeshHandle]:
    links = []
    for index, link_name in enumerate(model.link_names):  # type: ignore[attr-defined]
        mesh = model.link_mesh(link_name)  # type: ignore[attr-defined]
        links.append(
            scene.add_mesh_simple(
                f"{name}/links/{index:03d}",
                vertices=np.asarray(mesh["vertices"], dtype=np.float32),
                faces=np.asarray(mesh["faces"]),
                color=color,
            )
        )
    return links


def _update_rigid_links(model: BodyModel, handles: list[viser.MeshHandle], **forward_kwargs: object) -> None:
    transforms = np.asarray(model.forward_links(**forward_kwargs))  # type: ignore[attr-defined]
    transforms = _unbatched_transforms(transforms, "forward_links")
    wxyzs = SO3.conversions.from_rotmat_to_quat(transforms[:, :3, :3], convention="wxyz", xp=np)
    positions = transforms[:, :3, 3]
    for handle, wxyz, position in zip(handles, wxyzs, positions):
        handle.wxyz = wxyz
        position = np.asarray(position, dtype=np.float32)
        assert position.shape == (3,)
        handle.position = position


def _skeleton_positions(skeleton: object) -> Float[np.ndarray, "N 3"]:
    transforms = _unbatched_transforms(np.asarray(skeleton), "forward_skeleton")
    return transforms[:, :3, 3].astype(np.float32, copy=False)


def _parents(model: BodyModel) -> list[int]:
    parents = [int(parent) for parent in model.parents]
    if len(parents) != model.num_joints:
        raise ValueError(f"Expected {model.num_joints} parents, got {len(parents)}.")
    return parents


def _unbatched_vertices(values: Float[np.ndarray, "..."], name: str) -> Float[np.ndarray, "V 3"]:
    if values.ndim != 2 or values.shape[-1] != 3:
        raise ValueError(f"{name} must be unbatched with shape (N, 3), got {values.shape}.")
    return values


def _unbatched_transforms(values: Float[np.ndarray, "..."], name: str) -> Float[np.ndarray, "N 4 4"]:
    if values.ndim != 3 or values.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must be unbatched with shape (N, 4, 4), got {values.shape}.")
    return values


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
