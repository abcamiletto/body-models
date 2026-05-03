"""JAX backend for SOMA model using Flax NNX."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..anny.jax import ANNY
from ..base import BodyModel
from ..mhr.jax import MHR
from ..rotations import VALID_ROTATION_TYPES, RotationType
from ..smpl.jax import SMPL
from ..smplx.jax import SMPLX
from .io import (
    MODEL_TYPE_SPECS,
    compute_kinematic_fronts,
    get_identity_model_path,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    load_pose_correctives_weights,
    simplify_mesh,
)
import body_models.soma.kernels.jax as core

PathLike = Path | str

__all__ = ["SOMA"]


class SOMA(BodyModel, nnx.Module):
    """SOMA body model with JAX/Flax NNX backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)
    _identity_model: object

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.match_warp = match_warp
        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)
        corrective_weights = load_pose_correctives_weights(resolved_path)

        mean_full = data["mean"]
        shapedirs_full = data["shapedirs"]
        faces = data["faces"]
        skin_weights_full = data["skin_weights_full"]

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            self._vertex_map = nnx.Variable(jnp.asarray(np.asarray(vertex_map, dtype=np.int64)))
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.mean_full = nnx.Variable(jnp.asarray(mean_full))
        self.mean_active = nnx.Variable(jnp.asarray(mean_active))
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full))
        self.shapedirs_active = nnx.Variable(jnp.asarray(shapedirs_active))
        self.eigenvalues = nnx.Variable(jnp.asarray(data["eigenvalues"]))
        self.bind_shape_full = nnx.Variable(jnp.asarray(data["bind_shape"]))
        self.bind_pose_world = nnx.Variable(jnp.asarray(data["bind_pose_world"]))
        self.bind_pose_local = nnx.Variable(jnp.asarray(data["bind_pose_local"]))
        self.t_pose_world = nnx.Variable(jnp.asarray(data["t_pose_world"]))
        self.joint_regressor = nnx.Variable(jnp.asarray(data["joint_regressor"]))
        self.corrective_bindpose = nnx.Variable(jnp.asarray(corrective_weights["bindpose"]))
        self.corrective_W1 = nnx.Variable(jnp.asarray(corrective_weights["W1"]))
        self.corrective_W2_rows = nnx.Variable(jnp.asarray(corrective_weights["W2_rows"]))
        self.corrective_W2_cols = nnx.Variable(jnp.asarray(corrective_weights["W2_cols"]))
        self.corrective_W2_values = nnx.Variable(jnp.asarray(corrective_weights["W2_values"]))
        self._corrective_use_tanh = bool(corrective_weights["use_tanh"])
        self._skin_weights_full = nnx.Variable(jnp.asarray(skin_weights_full))
        self._skin_weights_active = nnx.Variable(jnp.asarray(skin_weights_active))
        self._faces = nnx.Variable(jnp.asarray(np.asarray(faces, dtype=np.int64)))
        self._identity_internal_to_source_rotation = nnx.Variable(jnp.eye(3, dtype=self.mean_full[...].dtype))
        self._identity_internal_to_source_translation = nnx.Variable(jnp.zeros(3, dtype=self.mean_full[...].dtype))
        self._identity_source_to_soma_rotation = nnx.Variable(jnp.eye(3, dtype=self.mean_full[...].dtype))

        self.parents = list(data["parents"])
        self._parents_full = data["joint_parents_full"].tolist()
        self._joint_children_full = data["joint_children_full"]
        self._skinned_vertex_indices_full = data["skinned_vertex_indices_full"]
        self._parents_full_index = nnx.Variable(jnp.asarray(self._parents_full))
        self._joint_children_indices_full = nnx.Variable(jnp.asarray(data["joint_children_indices_full"]))
        self._skinned_vertex_indices_full_index = nnx.Variable(jnp.asarray(data["skinned_vertex_indices_full_index"]))
        self._kinematic_fronts_full = compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data["joint_names"])

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source_scale = spec.source_scale
        self._identity_output_scale = spec.output_scale

        if spec.asset_dir is None:
            return

        transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
        self._identity_source_tetrahedra = nnx.Variable(jnp.asarray(transfer_data["source_tetrahedra"]))
        self._identity_face_ids = nnx.Variable(jnp.asarray(transfer_data["face_ids"]))
        self._identity_bary_coords = nnx.Variable(jnp.asarray(transfer_data["bary_coords"]))
        self._identity_unknown_ids = nnx.Variable(jnp.asarray(transfer_data["unknown_ids"]))
        self._identity_anchor_ids = nnx.Variable(jnp.asarray(transfer_data["anchor_ids"]))
        self._identity_solve_matrix = nnx.Variable(jnp.asarray(transfer_data["solve_matrix"]))
        self._identity_anchor_matrix = nnx.Variable(jnp.asarray(transfer_data["anchor_matrix"]))
        self._identity_rhs_base = nnx.Variable(jnp.asarray(transfer_data["rhs_base"]))
        self._identity_model = {
            "mhr": self._init_mhr_identity_backend,
            "anny": self._init_anny_identity_backend,
            "smpl": self._init_linear_identity_backend,
            "smplx": self._init_linear_identity_backend,
        }[self.model_type](transfer_data)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
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
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self._skin_weights_active[...][:, 1:]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.mean_active[...] * 0.01

    def forward_vertices(
        self,
        pose: Float[jax.Array, "B 77 N"] | Float[jax.Array, "B 77 3 3"],
        *,
        identity: Float[jax.Array, "B|1 I"] | None = None,
        scale_params: Float[jax.Array, "B|1 K"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> Float[jax.Array, "B V 3"]:
        identity, rest_shape_full, rest_shape_active = self._get_rest_shape(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.ops.forward_vertices(
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
            joint_children_indices_full=self._joint_children_indices_full[...],
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=self._skinned_vertex_indices_full_index[...],
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            parents_full_index=self._parents_full_index[...],
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
            match_warp=self.match_warp,
        )

    def forward_skeleton(
        self,
        pose: Float[jax.Array, "B 77 N"] | Float[jax.Array, "B 77 3 3"],
        *,
        identity: Float[jax.Array, "B|1 I"] | None = None,
        scale_params: Float[jax.Array, "B|1 K"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> Float[jax.Array, "B 77 4 4"]:
        identity, rest_shape_full, _rest_shape_active = self._get_rest_shape(
            identity=identity,
            scale_params=scale_params,
            ref=pose,
        )
        return core.ops.forward_skeleton(
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
            joint_children_indices_full=self._joint_children_indices_full[...],
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=self._skinned_vertex_indices_full_index[...],
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            parents_full_index=self._parents_full_index[...],
            identity=identity,
            pose=pose,
            rest_shape_full=rest_shape_full,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            match_warp=self.match_warp,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        params = {
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
        params["identity"] = jnp.full((1, self.identity_dim), self._default_identity_value, dtype=dtype)
        if self.num_scale_params is not None:
            params["scale_params"] = jnp.zeros((1, self.num_scale_params), dtype=dtype)
        return params

    def _get_rest_shape(
        self,
        *,
        identity: Float[jax.Array, "B|1 I"] | None,
        scale_params: Float[jax.Array, "B|1 K"] | None,
        ref: Float[jax.Array, "B ..."],
    ) -> tuple[
        Float[jax.Array, "B|1 I"] | None,
        Float[jax.Array, "B V 3"] | None,
        Float[jax.Array, "B V 3"] | None,
    ]:
        if identity is None:
            identity = jnp.full((1, self.identity_dim), self._default_identity_value, dtype=ref.dtype)
        identity, scale_params = core.ops.resolve_identity_inputs(
            identity=identity,
            scale_params=scale_params,
            batch_size=ref.shape[0],
            identity_dim=self.identity_dim,
            num_scale_params=self.num_scale_params,
            ref=ref,
            xp=jnp,
        )
        if self.model_type == "soma":
            return identity, None, None
        rest_shape, rest_shape_active = core.ops.prepare_identity_shape(
            model_type=self.model_type,
            identity_model=self._identity_model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=self.num_scale_params,
            identity_internal_to_source_rotation=self._identity_internal_to_source_rotation,
            identity_internal_to_source_translation=self._identity_internal_to_source_translation,
            identity_source_to_soma_rotation=self._identity_source_to_soma_rotation,
            identity_source_scale=self._identity_source_scale,
            identity_output_scale=self._identity_output_scale,
            identity_source_tetrahedra=self._identity_source_tetrahedra,
            identity_face_ids=self._identity_face_ids,
            identity_bary_coords=self._identity_bary_coords,
            identity_unknown_ids=self._identity_unknown_ids,
            identity_anchor_ids=self._identity_anchor_ids,
            identity_solve_matrix=self._identity_solve_matrix,
            identity_anchor_matrix=self._identity_anchor_matrix,
            identity_rhs_base=self._identity_rhs_base,
            vertex_map=self._vertex_map,
            xp=jnp,
        )
        return identity, rest_shape, rest_shape_active

    def _init_mhr_identity_backend(self, _transfer_data: dict[str, np.ndarray]) -> object:
        return nnx.data(MHR(model_path=get_identity_model_path("mhr"), simplify=1.0))

    def _init_anny_identity_backend(self, transfer_data: dict[str, np.ndarray]) -> object:
        identity_model = ANNY(
            model_path=get_identity_model_path("anny"),
            all_phenotypes=False,
            simplify=1.0,
        )
        source_vertices = jnp.asarray(transfer_data["source_vertices"])
        rotation, translation = core.ops.fit_rigid_transform(
            identity_model.template_vertices[...],
            source_vertices,
            xp=jnp,
        )
        self._identity_internal_to_source_rotation = nnx.Variable(rotation)
        self._identity_internal_to_source_translation = nnx.Variable(translation)
        self._identity_source_to_soma_rotation = nnx.Variable(
            jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        )
        return nnx.data(identity_model)

    def _init_linear_identity_backend(self, _transfer_data: dict[str, np.ndarray]) -> object:
        linear_model_cls = {"smpl": SMPL, "smplx": SMPLX}[self.model_type]
        return nnx.data(
            linear_model_cls(
                model_path=get_identity_model_path(self.model_type),
                simplify=1.0,
            )
        )
