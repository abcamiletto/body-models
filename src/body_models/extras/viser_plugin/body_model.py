"""Body model visualization helpers for ``viser``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float

from body_models.base import BodyModel, KinematicJoint
from nanomanifold import SO3

if TYPE_CHECKING:
    import viser


__all__ = [
    "ViserBodyModelHandle",
    "add_body_model",
]


class ViserBodyModelHandle:
    """Non-rigid body model rendered as a skinned mesh."""

    def __init__(
        self,
        model: BodyModel,
        pose: dict[str, Float[np.ndarray, "..."]],
        root_frame: viser.FrameHandle,
        mesh: viser.MeshSkinnedHandle,
    ) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self.pose = pose
        self.root_frame = root_frame
        self.mesh = mesh
        self.joint_handles: dict[int, viser.TransformControlsHandle] = {}
        self._syncing_joint_handles = False

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
        assert "shape" in self.pose, f"{self.model_name} does not support 'shape'."
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "shape" in self.pose, f"{self.model_name} does not support 'shape'."
        self.pose["shape"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=True)

    @property
    def body_pose(self) -> Float[np.ndarray, "..."]:
        assert "body_pose" in self.pose, f"{self.model_name} does not support 'body_pose'."
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "body_pose" in self.pose, f"{self.model_name} does not support 'body_pose'."
        self.pose["body_pose"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=False)

    @property
    def hand_pose(self) -> Float[np.ndarray, "..."]:
        assert "hand_pose" in self.pose, f"{self.model_name} does not support 'hand_pose'."
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "hand_pose" in self.pose, f"{self.model_name} does not support 'hand_pose'."
        self.pose["hand_pose"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=False)

    @property
    def head_pose(self) -> Float[np.ndarray, "..."]:
        assert "head_pose" in self.pose, f"{self.model_name} does not support 'head_pose'."
        return self.pose["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "head_pose" in self.pose, f"{self.model_name} does not support 'head_pose'."
        self.pose["head_pose"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=False)

    @property
    def expression(self) -> Float[np.ndarray, "..."]:
        assert "expression" in self.pose, f"{self.model_name} does not support 'expression'."
        return self.pose["expression"]

    @expression.setter
    def expression(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "expression" in self.pose, f"{self.model_name} does not support 'expression'."
        self.pose["expression"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=True)

    @property
    def global_rotation(self) -> Float[np.ndarray, "..."]:
        assert "global_rotation" in self.pose, f"{self.model_name} does not support 'global_rotation'."
        return self.pose["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_rotation" in self.pose, f"{self.model_name} does not support 'global_rotation'."
        self.pose["global_rotation"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=False)

    @property
    def global_translation(self) -> Float[np.ndarray, "..."]:
        assert "global_translation" in self.pose, f"{self.model_name} does not support 'global_translation'."
        return self.pose["global_translation"]

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "..."] | np.ndarray) -> None:
        assert "global_translation" in self.pose, f"{self.model_name} does not support 'global_translation'."
        self.pose["global_translation"] = np.asarray(value)
        self._apply_pose(rebuild_mesh=False)

    def set_pose(self, **forward_kwargs: Float[np.ndarray, "..."] | np.ndarray) -> None:
        rebuild_mesh = False
        changed = False
        for name, value in forward_kwargs.items():
            assert name in self.pose, f"{self.model_name} does not support {name!r}."
            value = np.asarray(value)
            if np.array_equal(self.pose[name], value):
                continue
            self.pose[name] = value.copy()
            changed = True
            rebuild_mesh = rebuild_mesh or name not in self.model.POSE_PARAMETER_NAMES
        if not changed:
            return
        self._apply_pose(rebuild_mesh=rebuild_mesh)

    def _apply_pose(self, *, rebuild_mesh: bool = False) -> None:
        if rebuild_mesh:
            self.mesh.vertices = self.model.to_viser_skinned_mesh(**self.pose)["vertices"]
        bones = self.model.to_viser_bones(**self.pose)
        for bone, wxyz, position in zip(
            self.mesh.bones,
            bones["bone_wxyzs"],
            bones["bone_positions"],
        ):
            bone.wxyz = wxyz
            bone.position = position
        self._sync_joint_handles_from_pose()

    def _sync_joint_handles_from_pose(self) -> None:
        if not self.joint_handles:
            return

        self._syncing_joint_handles = True
        skeleton = np.asarray(self.model.forward_skeleton(**self.pose))
        local = _local_joint_transforms(skeleton, self.model.parents)
        local_wxyzs = SO3.conversions.from_rotmat_to_quat(local[:, :3, :3], convention="wxyz", xp=np)
        for joint_index, handle in self.joint_handles.items():
            handle.wxyz = local_wxyzs[joint_index]
            handle.position = local[joint_index, :3, 3]
        self._syncing_joint_handles = False

    def _set_joint_pose_from_handle(self, joint: KinematicJoint, wxyz: np.ndarray) -> None:
        if self._syncing_joint_handles:
            return
        assert joint.rotation_parameter is not None

        rotmat = SO3.conversions.from_quat_to_rotmat(wxyz, convention="wxyz", xp=np)
        axis_angle = SO3.conversions.from_rotmat_to_axis_angle(rotmat, xp=np)
        rotation = SO3.convert(axis_angle, src="axis_angle", dst=getattr(self.model, "rotation_type"), xp=np)
        if joint.rotation_index is None:
            self.pose[joint.rotation_parameter] = rotation
        else:
            rotations = self.pose[joint.rotation_parameter].copy()
            rotations[joint.rotation_index] = rotation
            self.pose[joint.rotation_parameter] = rotations

        self._apply_pose(rebuild_mesh=False)

    def remove(self) -> None:
        for handle in self.joint_handles.values():
            handle.remove()
        self.mesh.remove()
        self.root_frame.remove()


def add_body_model(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
    joint_handles: bool = False,
) -> ViserBodyModelHandle:
    """Add a non-rigid body model to a ``viser`` scene."""
    if model.is_rigid_body:
        raise ValueError("add_body_model() only supports non-rigid models.")

    pose = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)
    mesh = model.to_viser_skinned_mesh(**pose)
    mesh_path = f"{name}/mesh"
    mesh_handle = scene.add_mesh_skinned(
        mesh_path,
        vertices=mesh["vertices"],
        faces=mesh["faces"],
        bone_wxyzs=mesh["bone_wxyzs"],
        bone_positions=mesh["bone_positions"],
        skin_weights=mesh["skin_weights"],
        color=color,
    )
    handle = ViserBodyModelHandle(model, pose, root, mesh_handle)
    if joint_handles:
        local = _local_joint_transforms(np.asarray(model.forward_skeleton(**pose)), model.parents)
        local_wxyzs = SO3.conversions.from_rotmat_to_quat(local[:, :3, :3], convention="wxyz", xp=np)

        prefixed_joint_names: list[str] = []
        for joint_index, joint in enumerate(model.kinematic_chain):
            joint_name = joint.name
            prefixed_joint_name = f"{joint_index}_{joint_name}"
            if joint_index > 0:
                prefixed_joint_name = f"{prefixed_joint_names[joint.parent]}/{prefixed_joint_name}"
            prefixed_joint_names.append(prefixed_joint_name)
            if joint.rotation_parameter is None:
                continue

            controls = scene.add_transform_controls(
                f"{name}/joint_handles/{prefixed_joint_name}",
                wxyz=local_wxyzs[joint_index],
                position=local[joint_index, :3, 3],
                depth_test=False,
                scale=0.2,
                disable_axes=True,
                disable_sliders=True,
            )
            handle.joint_handles[joint_index] = controls

            @controls.on_update
            def _(_, joint=joint, controls=controls) -> None:
                handle._set_joint_pose_from_handle(joint, np.asarray(controls.wxyz))

    return handle


def _local_joint_transforms(
    world: Float[np.ndarray, "J 4 4"],
    parents: list[int],
) -> Float[np.ndarray, "J 4 4"]:
    local = world.copy()
    for i, parent in enumerate(parents):
        if parent >= 0:
            local[i] = np.linalg.inv(world[parent]) @ world[i]
    return local
