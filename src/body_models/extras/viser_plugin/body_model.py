"""Body model visualization helpers for ``viser``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from body_models.base import BodyModel

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
            bind_pose = self.model.get_bind_params(**self.pose)
            self.mesh.vertices = _bind_vertices(self.model, bind_pose)

        pose = dict(self.pose)
        skeleton_wxyzs, skeleton_positions = _skeleton_poses(self.model, pose)
        for bone, wxyz, position in zip(self.mesh.bones, skeleton_wxyzs, skeleton_positions):
            bone.wxyz = wxyz
            bone.position = position

    def remove(self) -> None:
        self.mesh.remove()
        self.root_frame.remove()


def add_body_model(
    scene: viser.SceneApi,
    name: str,
    model: BodyModel,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
) -> ViserBodyModelHandle:
    """Add a non-rigid body model to a ``viser`` scene."""
    if model.is_rigid_body:
        raise ValueError("add_body_model() only supports non-rigid models.")

    pose = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)
    bind_pose = model.get_bind_params(**pose)
    skeleton_wxyzs, skeleton_positions = _skeleton_poses(model, bind_pose)
    mesh_path = f"{name}/mesh"
    mesh_handle = scene.add_mesh_skinned(
        mesh_path,
        vertices=_bind_vertices(model, bind_pose),
        faces=_triangular_faces(model),
        bone_wxyzs=skeleton_wxyzs,
        bone_positions=skeleton_positions,
        skin_weights=_viser_skin_weights(_numpy_array(model.skin_weights)),
        color=color,
    )
    return ViserBodyModelHandle(model, pose, root, mesh_handle)


def _skeleton_poses(
    model: BodyModel,
    forward_kwargs: dict[str, object],
) -> tuple[Float[np.ndarray, "J 4"], Float[np.ndarray, "J 3"]]:
    if "joint_indices" in forward_kwargs:
        raise ValueError("add_body_model() requires the full skeleton; do not pass joint_indices.")

    skeleton = _numpy_array(model.forward_skeleton(**forward_kwargs))
    if skeleton.ndim != 3 or skeleton.shape[-2:] != (4, 4):
        raise ValueError(f"add_body_model() expects unbatched skeleton shape (N, 4, 4), got {skeleton.shape}.")

    skeleton_wxyzs = SO3.conversions.from_rotmat_to_quat(skeleton[:, :3, :3], convention="wxyz", xp=np)
    skeleton_positions = skeleton[:, :3, 3].copy()
    return skeleton_wxyzs, skeleton_positions


def _bind_vertices(
    model: BodyModel,
    forward_kwargs: dict[str, object],
) -> Float[np.ndarray, "V 3"]:
    if "vertex_indices" in forward_kwargs:
        raise ValueError("add_body_model() requires the full mesh; do not pass vertex_indices.")

    vertices = _numpy_array(model.forward_vertices(**forward_kwargs))
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"add_body_model() expects unbatched vertices shape (N, 3), got {vertices.shape}.")
    return vertices.copy()


def _triangular_faces(model: BodyModel) -> np.ndarray:
    faces = _numpy_array(model.faces)
    if faces.shape[1] == 4:
        triangles_a = faces[:, [0, 1, 2]]
        triangles_b = faces[:, [0, 2, 3]]
        return np.concatenate([triangles_a, triangles_b], axis=0)
    if faces.shape[1] != 3:
        raise ValueError(f"Expected triangular or quad faces, got shape {faces.shape}.")
    return faces.copy()


def _viser_skin_weights(skin_weights: Float[np.ndarray, "V J"]) -> Float[np.ndarray, "V J"]:
    weights = np.asarray(skin_weights).copy()
    pruned_indices = np.argsort(weights, axis=-1)[:, :-4]
    weights[np.arange(weights.shape[0])[:, None], pruned_indices] = 0.0
    row_sums = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)


def _numpy_array(array: object) -> np.ndarray:
    detach = getattr(array, "detach", None)
    if callable(detach):
        array = detach()
    cpu = getattr(array, "cpu", None)
    if callable(cpu):
        array = cpu()
    return np.asarray(array)
