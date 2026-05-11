"""JAX backend for SOMA model."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .io import (
    MODEL_TYPE_SPECS,
    compute_sparse_skin_weights,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
from body_models.soma.backends import jax as backend
from body_models.soma.backends import core
from body_models.soma import identities
from body_models.soma.identities import jax as identity_sources
from body_models.soma.constants import SOMA_APOSE, SOMA_IPOSE, SOMA_JOINTS

PathLike = Path | str

__all__ = ["SOMA"]


class SOMA(BodyModel):
    """SOMA body model with JAX backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)
    JOINTS = SOMA_JOINTS

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
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0 (1.0 = original mesh)")

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.match_warp = match_warp
        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        mean_full = data.mean_full
        shapedirs_full = data.shapedirs_full
        faces = data.faces
        skin_weights_full = data.skin_weights_full

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            vertex_map = np.asarray(vertex_map, dtype=np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            vertex_map = None

        skin_joint_indices_active, skin_joint_weights_active = compute_sparse_skin_weights(skin_weights_active)
        weights = replace(
            data,
            mean_active=np.asarray(mean_active, dtype=np.float32),
            shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
            skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
            skin_joint_indices_active=skin_joint_indices_active,
            skin_joint_weights_active=skin_joint_weights_active,
            faces=np.asarray(faces, dtype=np.int64),
            vertex_map=vertex_map,
        )
        self.weights = common.jaxify(weights)

        self.parents = [parent - 1 for parent in data.topology.parents_full[1:]]
        self._joint_names = data.joint_names_full[1:]

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
            self._identity_source = identity_sources.create_identity_source(self.model_type, transfer_data)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self.weights.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.mean_active * 0.01

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
        prepared_identity: core.PreparedSomaIdentity | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        identity_state = prepared_identity
        if identity_state is None:
            identity_state = self.prepare_identity(identity=identity, scale_params=scale_params, pose=pose)
        return backend.forward_vertices(
            data=self.weights,
            prepared_identity=identity_state,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            xp=jnp,
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
        prepared_identity: core.PreparedSomaIdentity | None = None,
    ) -> Float[jax.Array, "B 77 4 4"]:
        identity_state = prepared_identity
        if identity_state is None:
            identity_state = self.prepare_identity(identity=identity, scale_params=scale_params, pose=pose)
        return backend.forward_skeleton(
            data=self.weights,
            prepared_identity=identity_state,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            xp=jnp,
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
        params["identity"] = jnp.full(
            (1, self.identity_dim),
            self._default_identity_value,
            dtype=dtype,
        )
        if self.num_scale_params is not None:
            params["scale_params"] = jnp.zeros((1, self.num_scale_params), dtype=dtype)
        return params

    def prepare_identity(
        self,
        *,
        identity: Float[jax.Array, "B|1 I"] | None = None,
        scale_params: Float[jax.Array, "B|1 K"] | None = None,
        pose: Float[jax.Array, "B ..."],
    ) -> core.PreparedSomaIdentity:
        identity, scale_params = self._identity_inputs(identity=identity, scale_params=scale_params, pose=pose)
        return self._prepare_identity_from_inputs(identity, scale_params)

    def _identity_inputs(
        self,
        *,
        identity: Float[jax.Array, "B|1 I"] | None,
        scale_params: Float[jax.Array, "B|1 K"] | None,
        pose: Float[jax.Array, "B ..."],
    ) -> tuple[Float[jax.Array, "B I"], Float[jax.Array, "B K"] | None]:
        batch_size = pose.shape[0]
        if identity is None:
            identity = jnp.full(
                (batch_size, self.identity_dim),
                self._default_identity_value,
                dtype=pose.dtype,
            )
        elif identity.shape[0] == 1 and batch_size > 1:
            identity = jnp.broadcast_to(identity, (batch_size, identity.shape[-1]))

        if self.num_scale_params is None:
            if scale_params is not None:
                raise ValueError("scale_params is only supported for SOMA model_type='mhr'.")
            return identity, None

        if scale_params is None:
            scale_params = jnp.zeros((batch_size, self.num_scale_params), dtype=identity.dtype)
        elif scale_params.shape[0] == 1 and batch_size > 1:
            scale_params = jnp.broadcast_to(scale_params, (batch_size, scale_params.shape[-1]))
        return identity, scale_params

    def _prepare_identity_from_inputs(
        self,
        identity: Float[jax.Array, "B I"],
        scale_params: Float[jax.Array, "B K"] | None,
    ) -> core.PreparedSomaIdentity:
        rest_shape_full, rest_shape_active = identities.rest_shapes(
            data=self.weights,
            identity_source=self._identity_source,
            identity=identity,
            scale_params=scale_params,
            xp=jnp,
        )
        return backend.prepare_identity_from_rest_shape(
            data=self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=jnp,
        )

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["pose"]
        for index, values in SOMA_APOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["pose"]
        for index, values in SOMA_IPOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["pose"] = body_pose
        return params
