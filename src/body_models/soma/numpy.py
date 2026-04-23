"""NumPy backend for SOMA model."""

from pathlib import Path as _Path

import numpy as _np
from jaxtyping import Float as _Float, Int as _Int
from nanomanifold import SO3 as _SO3

from ..base import BodyModel as _BodyModel
from ..rotations import VALID_ROTATION_TYPES as _VALID_ROTATION_TYPES
from . import core as _core
from .io import (
    compute_kinematic_fronts as _compute_kinematic_fronts,
    get_model_path as _get_model_path,
    load_model_data as _load_model_data,
    load_pose_correctives_weights as _load_pose_correctives_weights,
    simplify_mesh as _simplify_mesh,
)

__all__ = ["SOMA"]


class SOMA(_BodyModel):
    """SOMA body model with NumPy backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77

    def __init__(
        self,
        model_path: _Path | str | None = None,
        *,
        simplify: float = 1.0,
        rotation_type: _core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in _VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"

        self.rotation_type = rotation_type
        resolved_path = _get_model_path(model_path)
        data = _load_model_data(resolved_path)
        corrective_weights = _load_pose_correctives_weights(resolved_path)

        mean_full = data["mean"]
        shapedirs_full = data["shapedirs"]
        faces = data["faces"]
        skin_weights_full = data["skin_weights_full"]

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = _simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            self._vertex_map = _np.asarray(vertex_map, dtype=_np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.mean_full = mean_full
        self.mean_active = _np.asarray(mean_active, dtype=_np.float32)
        self.shapedirs_full = shapedirs_full
        self.shapedirs_active = _np.asarray(shapedirs_active, dtype=_np.float32)
        self.eigenvalues = data["eigenvalues"]
        self.bind_shape_full = data["bind_shape"]
        self.bind_pose_world = data["bind_pose_world"]
        self.bind_pose_local = data["bind_pose_local"]
        self.t_pose_world = data["t_pose_world"]
        self.joint_regressor = data["joint_regressor"]
        self.corrective_bindpose = _np.asarray(corrective_weights["bindpose"], dtype=_np.float32)
        self.corrective_W1 = _np.asarray(corrective_weights["W1"], dtype=_np.float32)
        self.corrective_W2_rows = _np.asarray(corrective_weights["W2_rows"], dtype=_np.int64)
        self.corrective_W2_cols = _np.asarray(corrective_weights["W2_cols"], dtype=_np.int64)
        self.corrective_W2_values = _np.asarray(corrective_weights["W2_values"], dtype=_np.float32)
        self._corrective_use_tanh = bool(corrective_weights["use_tanh"])
        self._skin_weights_full = skin_weights_full
        self._skin_weights_active = _np.asarray(skin_weights_active, dtype=_np.float32)
        self._faces = _np.asarray(faces, dtype=_np.int64)

        self.parents = list(data["parents"])
        self._parents_full = data["joint_parents_full"].tolist()
        self._joint_children_full = data["joint_children_full"]
        self._skinned_vertex_indices_full = data["skinned_vertex_indices_full"]
        self._kinematic_fronts_full = _compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data["joint_names"])

    @property
    def faces(self) -> _Int[_np.ndarray, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_active.shape[0]

    @property
    def skin_weights(self) -> _Float[_np.ndarray, "V J"]:
        return self._skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> _Float[_np.ndarray, "V 3"]:
        return self.mean_active * 0.01

    def forward_vertices(
        self,
        shape: _Float[_np.ndarray, "B|1 128"],
        pose: _Float[_np.ndarray, "B 77 N"] | _Float[_np.ndarray, "B 77 3 3"],
        global_rotation: _Float[_np.ndarray, "B N"] | _Float[_np.ndarray, "B 3 3"] | None = None,
        global_translation: _Float[_np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_np.ndarray, "B V 3"]:
        return _core.forward_vertices(
            mean_full=self.mean_full,
            mean_active=self.mean_active,
            shapedirs_full=self.shapedirs_full,
            shapedirs_active=self.shapedirs_active,
            eigenvalues=self.eigenvalues,
            bind_shape_full=self.bind_shape_full,
            skin_weights_active=self._skin_weights_active,
            bind_pose_world=self.bind_pose_world,
            bind_pose_local=self.bind_pose_local,
            t_pose_world=self.t_pose_world,
            joint_regressor=self.joint_regressor,
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            vertex_map=self._vertex_map,
            corrective_bindpose=self.corrective_bindpose,
            corrective_W1=self.corrective_W1,
            corrective_W2_rows=self.corrective_W2_rows,
            corrective_W2_cols=self.corrective_W2_cols,
            corrective_W2_values=self.corrective_W2_values,
            corrective_use_tanh=self._corrective_use_tanh,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: _Float[_np.ndarray, "B|1 128"],
        pose: _Float[_np.ndarray, "B 77 N"] | _Float[_np.ndarray, "B 77 3 3"],
        global_rotation: _Float[_np.ndarray, "B N"] | _Float[_np.ndarray, "B 3 3"] | None = None,
        global_translation: _Float[_np.ndarray, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_np.ndarray, "B 77 4 4"]:
        return _core.forward_skeleton(
            mean_full=self.mean_full,
            shapedirs_full=self.shapedirs_full,
            eigenvalues=self.eigenvalues,
            bind_shape_full=self.bind_shape_full,
            skin_weights_full=self._skin_weights_full,
            bind_pose_world=self.bind_pose_world,
            bind_pose_local=self.bind_pose_local,
            t_pose_world=self.t_pose_world,
            joint_regressor=self.joint_regressor,
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=_np.float32) -> dict[str, _np.ndarray]:
        pose_ref = _np.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = _np.zeros((batch_size, 3), dtype=dtype)
        return {
            "shape": _np.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": _SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=_np,
            ),
            "global_rotation": _SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=_np,
            ),
            "global_translation": _np.zeros((batch_size, 3), dtype=dtype),
        }
