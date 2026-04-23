"""PyTorch backend for SOMA model."""

from pathlib import Path as _Path

import numpy as _np
import torch as _torch
import torch.nn as _nn
from jaxtyping import Float as _Float, Int as _Int
from nanomanifold import SO3 as _SO3
from torch import Tensor as _Tensor

from ..base import BodyModel as _BodyModel
from ..mhr.torch import MHR as _MHR
from ..rotations import VALID_ROTATION_TYPES as _VALID_ROTATION_TYPES
from ..smpl.torch import SMPL as _SMPL
from ..smplx.torch import SMPLX as _SMPLX
from . import core as _core
from .io import (
    IDENTITY_MODEL_TYPES as _IDENTITY_MODEL_TYPES,
    compute_kinematic_fronts as _compute_kinematic_fronts,
    get_identity_model_path as _get_identity_model_path,
    get_model_path as _get_model_path,
    load_identity_transfer_data as _load_identity_transfer_data,
    load_model_data as _load_model_data,
    load_pose_correctives_weights as _load_pose_correctives_weights,
    simplify_mesh as _simplify_mesh,
)

__all__ = ["SOMA"]


class SOMA(_BodyModel, _nn.Module):
    """SOMA body model with PyTorch backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = ("soma", *_IDENTITY_MODEL_TYPES)

    mean_full: _Tensor
    mean_active: _Tensor
    shapedirs_full: _Tensor
    shapedirs_active: _Tensor
    eigenvalues: _Tensor
    bind_shape_full: _Tensor
    bind_pose_world: _Tensor
    bind_pose_local: _Tensor
    t_pose_world: _Tensor
    joint_regressor: _Tensor
    corrective_bindpose: _Tensor
    corrective_W1: _Tensor
    corrective_W2_rows: _Tensor
    corrective_W2_cols: _Tensor
    corrective_W2_values: _Tensor
    _skin_weights_full: _Tensor
    _skin_weights_active: _Tensor
    _faces: _Tensor
    _vertex_map: _Tensor | None
    _identity_source_tetrahedra: _Tensor
    _identity_face_ids: _Tensor
    _identity_bary_coords: _Tensor
    _identity_unknown_ids: _Tensor
    _identity_anchor_ids: _Tensor
    _identity_solve_matrix: _Tensor
    _identity_anchor_matrix: _Tensor
    _identity_rhs_base: _Tensor
    _identity_mhr_model: _MHR | None
    _identity_linear_model: _SMPL | _SMPLX | None

    def __init__(
        self,
        model_path: _Path | str | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: _core.RotationType = "axis_angle",
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in _VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

        self.model_type = normalized_model_type
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
            self.register_buffer("_vertex_map", _torch.as_tensor(_np.asarray(vertex_map, dtype=_np.int64)))
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.register_buffer("mean_full", _torch.as_tensor(mean_full))
        self.register_buffer("mean_active", _torch.as_tensor(mean_active))
        self.register_buffer("shapedirs_full", _torch.as_tensor(shapedirs_full))
        self.register_buffer("shapedirs_active", _torch.as_tensor(shapedirs_active))
        self.register_buffer("eigenvalues", _torch.as_tensor(data["eigenvalues"]))
        self.register_buffer("bind_shape_full", _torch.as_tensor(data["bind_shape"]))
        self.register_buffer("bind_pose_world", _torch.as_tensor(data["bind_pose_world"]))
        self.register_buffer("bind_pose_local", _torch.as_tensor(data["bind_pose_local"]))
        self.register_buffer("t_pose_world", _torch.as_tensor(data["t_pose_world"]))
        self.register_buffer("joint_regressor", _torch.as_tensor(data["joint_regressor"]))
        self.register_buffer("corrective_bindpose", _torch.as_tensor(corrective_weights["bindpose"]))
        self.register_buffer("corrective_W1", _torch.as_tensor(corrective_weights["W1"]))
        self.register_buffer("corrective_W2_rows", _torch.as_tensor(corrective_weights["W2_rows"], dtype=_torch.int64))
        self.register_buffer("corrective_W2_cols", _torch.as_tensor(corrective_weights["W2_cols"], dtype=_torch.int64))
        self.register_buffer("corrective_W2_values", _torch.as_tensor(corrective_weights["W2_values"]))
        self.register_buffer("_skin_weights_full", _torch.as_tensor(skin_weights_full))
        self.register_buffer("_skin_weights_active", _torch.as_tensor(skin_weights_active))
        self.register_buffer("_faces", _torch.as_tensor(_np.asarray(faces, dtype=_np.int64)))

        self._corrective_use_tanh = bool(corrective_weights["use_tanh"])
        self.parents = list(data["parents"])
        self._parents_full = data["joint_parents_full"].tolist()
        self._joint_children_full = data["joint_children_full"]
        self._skinned_vertex_indices_full = data["skinned_vertex_indices_full"]
        self._kinematic_fronts_full = _compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data["joint_names"])

        self._identity_mhr_model = None
        self._identity_linear_model = None
        self._identity_source_scale = 1.0
        self._identity_output_scale = 1.0

        if self.model_type == "soma":
            self.identity_dim = self.SHAPE_DIM
            self.num_scale_params = None
            return

        transfer_data = _load_identity_transfer_data(resolved_path, self.model_type)
        self.register_buffer(
            "_identity_source_tetrahedra",
            _torch.as_tensor(transfer_data["source_tetrahedra"], dtype=_torch.int64),
        )
        self.register_buffer("_identity_face_ids", _torch.as_tensor(transfer_data["face_ids"], dtype=_torch.int64))
        self.register_buffer("_identity_bary_coords", _torch.as_tensor(transfer_data["bary_coords"]))
        self.register_buffer(
            "_identity_unknown_ids", _torch.as_tensor(transfer_data["unknown_ids"], dtype=_torch.int64)
        )
        self.register_buffer("_identity_anchor_ids", _torch.as_tensor(transfer_data["anchor_ids"], dtype=_torch.int64))
        self.register_buffer("_identity_solve_matrix", _torch.as_tensor(transfer_data["solve_matrix"]))
        self.register_buffer("_identity_anchor_matrix", _torch.as_tensor(transfer_data["anchor_matrix"]))
        self.register_buffer("_identity_rhs_base", _torch.as_tensor(transfer_data["rhs_base"]))

        if self.model_type == "mhr":
            self.identity_dim = 45
            self.num_scale_params = 68
            self._identity_source_scale = 100.0
            self._identity_mhr_model = _MHR(model_path=_get_identity_model_path("mhr"), simplify=1.0)
            return

        self.identity_dim = 10
        self.num_scale_params = None
        self._identity_output_scale = 100.0
        if self.model_type == "smplx":
            self._identity_linear_model = _SMPLX(
                model_path=_get_identity_model_path("smplx"),
                gender="neutral",
                simplify=1.0,
            )
        else:
            self._identity_linear_model = _SMPL(
                model_path=_get_identity_model_path("smpl"),
                gender="neutral",
                simplify=1.0,
            )

    @property
    def faces(self) -> _Int[_Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return int(self.mean_active.shape[0])

    @property
    def skin_weights(self) -> _Float[_Tensor, "V J"]:
        return self._skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> _Float[_Tensor, "V 3"]:
        return self.mean_active * 0.01

    def forward_vertices(
        self,
        pose: _Float[_Tensor, "B 77 N"] | _Float[_Tensor, "B 77 3 3"],
        *,
        shape: _Float[_Tensor, "B|1 128"] | None = None,
        identity: _Float[_Tensor, "B|1 I"] | None = None,
        scale_params: _Float[_Tensor, "B|1 K"] | None = None,
        global_rotation: _Float[_Tensor, "B N"] | _Float[_Tensor, "B 3 3"] | None = None,
        global_translation: _Float[_Tensor, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_Tensor, "B V 3"]:
        shape, rest_shape_full, rest_shape_active = self._resolve_identity_inputs(
            shape=shape,
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
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
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
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
            xp=_torch,
        )

    def forward_skeleton(
        self,
        pose: _Float[_Tensor, "B 77 N"] | _Float[_Tensor, "B 77 3 3"],
        *,
        shape: _Float[_Tensor, "B|1 128"] | None = None,
        identity: _Float[_Tensor, "B|1 I"] | None = None,
        scale_params: _Float[_Tensor, "B|1 K"] | None = None,
        global_rotation: _Float[_Tensor, "B N"] | _Float[_Tensor, "B 3 3"] | None = None,
        global_translation: _Float[_Tensor, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_Tensor, "B 77 4 4"]:
        shape, rest_shape_full, _rest_shape_active = self._resolve_identity_inputs(
            shape=shape,
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
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
            rest_shape_full=rest_shape_full,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            xp=_torch,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: _torch.dtype = _torch.float32) -> dict[str, _Tensor]:
        device = self.mean_active.device
        pose_ref = _torch.zeros((batch_size, self.num_joints, 3), device=device, dtype=dtype)
        rot_ref = _torch.zeros((batch_size, 3), device=device, dtype=dtype)
        params = {
            "pose": _SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=_torch,
            ),
            "global_rotation": _SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=_torch,
            ),
            "global_translation": _torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
        if self.model_type == "soma":
            params["shape"] = _torch.zeros((1, self.SHAPE_DIM), device=device, dtype=dtype)
            return params

        params["identity"] = _torch.zeros((1, self.identity_dim), device=device, dtype=dtype)
        if self.num_scale_params is not None:
            params["scale_params"] = _torch.zeros((1, self.num_scale_params), device=device, dtype=dtype)
        return params

    def _identity_rest_shape(
        self,
        identity: _Tensor,
        scale_params: _Tensor | None,
    ) -> _Tensor:
        if self.model_type == "mhr":
            model = self._identity_mhr_model
            if model is None:
                raise RuntimeError("Missing MHR identity backend for SOMA.")
            rest_shape = _core.mhr_identity_shape(
                model=model,
                identity=identity,
                scale_params=scale_params,
                num_scale_params=self.num_scale_params or 0,
                xp=_torch,
            )
        else:
            model = self._identity_linear_model
            if model is None:
                raise RuntimeError("Missing linear identity backend for SOMA.")
            rest_shape = _core.linear_identity_shape(
                mean=model.v_template_full,
                shapedirs=model.shapedirs_full,
                identity=identity,
                xp=_torch,
            )

        if self._identity_source_scale != 1.0:
            rest_shape = rest_shape * self._identity_source_scale

        rest_shape = _core.transfer_identity_rest_shape(
            source_shape=rest_shape,
            source_tetrahedra=self._identity_source_tetrahedra,
            face_ids=self._identity_face_ids,
            bary_coords=self._identity_bary_coords,
            unknown_ids=self._identity_unknown_ids,
            anchor_ids=self._identity_anchor_ids,
            solve_matrix=self._identity_solve_matrix,
            anchor_matrix=self._identity_anchor_matrix,
            rhs_base=self._identity_rhs_base,
            xp=_torch,
        )
        if self._identity_output_scale != 1.0:
            rest_shape = rest_shape * self._identity_output_scale
        return rest_shape

    def _resolve_identity_inputs(
        self,
        *,
        shape: _Tensor | None,
        identity: _Tensor | None,
        scale_params: _Tensor | None,
        ref: _Tensor,
    ) -> tuple[_Tensor | None, _Tensor | None, _Tensor | None]:
        shape, identity, scale_params = _core.resolve_identity_inputs(
            model_type=self.model_type,
            shape=shape,
            identity=identity,
            scale_params=scale_params,
            batch_size=ref.shape[0],
            identity_dim=self.identity_dim,
            num_scale_params=self.num_scale_params,
            ref=ref,
            xp=_torch,
        )
        if identity is None:
            return shape, None, None

        rest_shape_full = self._identity_rest_shape(identity, scale_params)
        if self._vertex_map is None:
            rest_shape_active = rest_shape_full
        else:
            rest_shape_active = rest_shape_full[:, self._vertex_map]
        return None, rest_shape_full, rest_shape_active
