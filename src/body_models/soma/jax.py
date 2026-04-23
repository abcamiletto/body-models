"""JAX backend for SOMA model using Flax NNX."""

from pathlib import Path as _Path
from typing import cast as _cast

import jax as _jax
import jax.numpy as _jnp
import numpy as _np
from flax import nnx as _nnx
from jaxtyping import Float as _Float, Int as _Int
from nanomanifold import SO3 as _SO3

from ..base import BodyModel as _BodyModel
from ..mhr.jax import MHR as _MHR
from ..rotations import VALID_ROTATION_TYPES as _VALID_ROTATION_TYPES
from ..smpl.jax import SMPL as _SMPL
from ..smplx.jax import SMPLX as _SMPLX
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


class SOMA(_BodyModel, _nnx.Module):
    """SOMA body model with JAX/Flax NNX backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = ("soma", *_IDENTITY_MODEL_TYPES)
    _identity_mhr_model: _MHR
    _identity_linear_model: _SMPL | _SMPLX

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
            self._vertex_map = _nnx.Variable(_jnp.asarray(_np.asarray(vertex_map, dtype=_np.int64)))
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.mean_full = _nnx.Variable(_jnp.asarray(mean_full))
        self.mean_active = _nnx.Variable(_jnp.asarray(mean_active))
        self.shapedirs_full = _nnx.Variable(_jnp.asarray(shapedirs_full))
        self.shapedirs_active = _nnx.Variable(_jnp.asarray(shapedirs_active))
        self.eigenvalues = _nnx.Variable(_jnp.asarray(data["eigenvalues"]))
        self.bind_shape_full = _nnx.Variable(_jnp.asarray(data["bind_shape"]))
        self.bind_pose_world = _nnx.Variable(_jnp.asarray(data["bind_pose_world"]))
        self.bind_pose_local = _nnx.Variable(_jnp.asarray(data["bind_pose_local"]))
        self.t_pose_world = _nnx.Variable(_jnp.asarray(data["t_pose_world"]))
        self.joint_regressor = _nnx.Variable(_jnp.asarray(data["joint_regressor"]))
        self.corrective_bindpose = _nnx.Variable(_jnp.asarray(corrective_weights["bindpose"]))
        self.corrective_W1 = _nnx.Variable(_jnp.asarray(corrective_weights["W1"]))
        self.corrective_W2_rows = _nnx.Variable(_jnp.asarray(corrective_weights["W2_rows"]))
        self.corrective_W2_cols = _nnx.Variable(_jnp.asarray(corrective_weights["W2_cols"]))
        self.corrective_W2_values = _nnx.Variable(_jnp.asarray(corrective_weights["W2_values"]))
        self._corrective_use_tanh = bool(corrective_weights["use_tanh"])
        self._skin_weights_full = _nnx.Variable(_jnp.asarray(skin_weights_full))
        self._skin_weights_active = _nnx.Variable(_jnp.asarray(skin_weights_active))
        self._faces = _nnx.Variable(_jnp.asarray(_np.asarray(faces, dtype=_np.int64)))

        self.parents = list(data["parents"])
        self._parents_full = data["joint_parents_full"].tolist()
        self._joint_children_full = data["joint_children_full"]
        self._skinned_vertex_indices_full = data["skinned_vertex_indices_full"]
        self._kinematic_fronts_full = _compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data["joint_names"])

        self._identity_source_scale = 1.0
        self._identity_output_scale = 1.0

        if self.model_type == "soma":
            self.identity_dim = self.SHAPE_DIM
            self.num_scale_params = None
            return

        transfer_data = _load_identity_transfer_data(resolved_path, self.model_type)
        self._identity_source_tetrahedra = _nnx.Variable(_jnp.asarray(transfer_data["source_tetrahedra"]))
        self._identity_face_ids = _nnx.Variable(_jnp.asarray(transfer_data["face_ids"]))
        self._identity_bary_coords = _nnx.Variable(_jnp.asarray(transfer_data["bary_coords"]))
        self._identity_unknown_ids = _nnx.Variable(_jnp.asarray(transfer_data["unknown_ids"]))
        self._identity_anchor_ids = _nnx.Variable(_jnp.asarray(transfer_data["anchor_ids"]))
        self._identity_solve_matrix = _nnx.Variable(_jnp.asarray(transfer_data["solve_matrix"]))
        self._identity_anchor_matrix = _nnx.Variable(_jnp.asarray(transfer_data["anchor_matrix"]))
        self._identity_rhs_base = _nnx.Variable(_jnp.asarray(transfer_data["rhs_base"]))

        if self.model_type == "mhr":
            self.identity_dim = 45
            self.num_scale_params = 68
            self._identity_source_scale = 100.0
            self._identity_mhr_model = _nnx.data(_MHR(model_path=_get_identity_model_path("mhr"), simplify=1.0))
            return

        self.identity_dim = 10
        self.num_scale_params = None
        self._identity_output_scale = 100.0
        if self.model_type == "smplx":
            self._identity_linear_model = _nnx.data(
                _SMPLX(
                    model_path=_get_identity_model_path("smplx"),
                    gender="neutral",
                    simplify=1.0,
                )
            )
        else:
            self._identity_linear_model = _nnx.data(
                _SMPL(
                    model_path=_get_identity_model_path("smpl"),
                    gender="neutral",
                    simplify=1.0,
                )
            )

    @property
    def faces(self) -> _Int[_jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_active[...].shape[0]

    @property
    def skin_weights(self) -> _Float[_jax.Array, "V J"]:
        return self._skin_weights_active[...][:, 1:]

    @property
    def rest_vertices(self) -> _Float[_jax.Array, "V 3"]:
        return self.mean_active[...] * 0.01

    def forward_vertices(
        self,
        pose: _Float[_jax.Array, "B 77 N"] | _Float[_jax.Array, "B 77 3 3"],
        *,
        identity: _Float[_jax.Array, "B|1 I"] | None = None,
        scale_params: _Float[_jax.Array, "B|1 K"] | None = None,
        global_rotation: _Float[_jax.Array, "B N"] | _Float[_jax.Array, "B 3 3"] | None = None,
        global_translation: _Float[_jax.Array, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_jax.Array, "B V 3"]:
        identity, rest_shape_full, rest_shape_active = self._resolve_identity_inputs(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return _core.forward_vertices(
            mean_full=self.mean_full[...],
            mean_active=self.mean_active[...],
            shapedirs_full=self.shapedirs_full[...],
            shapedirs_active=self.shapedirs_active[...],
            eigenvalues=self.eigenvalues[...],
            bind_shape_full=self.bind_shape_full[...],
            skin_weights_active=self._skin_weights_active[...],
            bind_pose_world=self.bind_pose_world[...],
            bind_pose_local=self.bind_pose_local[...],
            t_pose_world=self.t_pose_world[...],
            joint_regressor=self.joint_regressor[...],
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            vertex_map=None if self._vertex_map is None else self._vertex_map[...],
            corrective_bindpose=self.corrective_bindpose[...],
            corrective_W1=self.corrective_W1[...],
            corrective_W2_rows=self.corrective_W2_rows[...],
            corrective_W2_cols=self.corrective_W2_cols[...],
            corrective_W2_values=self.corrective_W2_values[...],
            corrective_use_tanh=self._corrective_use_tanh,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        pose: _Float[_jax.Array, "B 77 N"] | _Float[_jax.Array, "B 77 3 3"],
        *,
        identity: _Float[_jax.Array, "B|1 I"] | None = None,
        scale_params: _Float[_jax.Array, "B|1 K"] | None = None,
        global_rotation: _Float[_jax.Array, "B N"] | _Float[_jax.Array, "B 3 3"] | None = None,
        global_translation: _Float[_jax.Array, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> _Float[_jax.Array, "B 77 4 4"]:
        identity, rest_shape_full, _rest_shape_active = self._resolve_identity_inputs(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return _core.forward_skeleton(
            mean_full=self.mean_full[...],
            shapedirs_full=self.shapedirs_full[...],
            eigenvalues=self.eigenvalues[...],
            bind_shape_full=self.bind_shape_full[...],
            skin_weights_full=self._skin_weights_full[...],
            bind_pose_world=self.bind_pose_world[...],
            bind_pose_local=self.bind_pose_local[...],
            t_pose_world=self.t_pose_world[...],
            joint_regressor=self.joint_regressor[...],
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=_jnp.float32) -> dict[str, _jax.Array]:
        pose_ref = _jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = _jnp.zeros((batch_size, 3), dtype=dtype)
        params = {
            "pose": _SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=_jnp,
            ),
            "global_rotation": _SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=_jnp,
            ),
            "global_translation": _jnp.zeros((batch_size, 3), dtype=dtype),
        }
        params["identity"] = _jnp.zeros((1, self.identity_dim), dtype=dtype)
        if self.num_scale_params is not None:
            params["scale_params"] = _jnp.zeros((1, self.num_scale_params), dtype=dtype)
        return params

    def _identity_rest_shape(
        self,
        identity: _Float[_jax.Array, "B I"],
        scale_params: _Float[_jax.Array, "B K"] | None,
    ) -> _Float[_jax.Array, "B V 3"]:
        if self.model_type == "mhr":
            num_scale_params = _cast(int, self.num_scale_params)
            rest_shape = _core.mhr_identity_shape(
                model=self._identity_mhr_model,
                identity=identity,
                scale_params=scale_params,
                num_scale_params=num_scale_params,
                xp=_jnp,
            )
        else:
            rest_shape = _core.linear_identity_shape(
                mean=self._identity_linear_model.v_template_full[...],
                shapedirs=self._identity_linear_model.shapedirs_full[...],
                identity=identity,
                xp=_jnp,
            )

        if self._identity_source_scale != 1.0:
            rest_shape = rest_shape * self._identity_source_scale

        rest_shape = _core.transfer_identity_rest_shape(
            source_shape=rest_shape,
            source_tetrahedra=self._identity_source_tetrahedra[...],
            face_ids=self._identity_face_ids[...],
            bary_coords=self._identity_bary_coords[...],
            unknown_ids=self._identity_unknown_ids[...],
            anchor_ids=self._identity_anchor_ids[...],
            solve_matrix=self._identity_solve_matrix[...],
            anchor_matrix=self._identity_anchor_matrix[...],
            rhs_base=self._identity_rhs_base[...],
            xp=_jnp,
        )
        if self._identity_output_scale != 1.0:
            rest_shape = rest_shape * self._identity_output_scale
        return rest_shape

    def _resolve_identity_inputs(
        self,
        *,
        identity: _Float[_jax.Array, "B|1 I"] | None,
        scale_params: _Float[_jax.Array, "B|1 K"] | None,
        ref: _Float[_jax.Array, "B ..."],
    ) -> tuple[
        _Float[_jax.Array, "B|1 I"] | None,
        _Float[_jax.Array, "B V 3"] | None,
        _Float[_jax.Array, "B V 3"] | None,
    ]:
        identity, scale_params = _core.resolve_identity_inputs(
            identity=identity,
            scale_params=scale_params,
            batch_size=ref.shape[0],
            identity_dim=self.identity_dim,
            num_scale_params=self.num_scale_params,
            ref=ref,
            xp=_jnp,
        )
        if self.model_type == "soma":
            return identity, None, None

        rest_shape_full = self._identity_rest_shape(identity, scale_params)
        if self._vertex_map is None:
            rest_shape_active = rest_shape_full
        else:
            rest_shape_active = rest_shape_full[:, self._vertex_map[...]]
        return None, rest_shape_full, rest_shape_active
