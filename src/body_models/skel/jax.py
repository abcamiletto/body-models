"""JAX backend for SKEL model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from body_models.skel.backends import jax as backend
from body_models.skel.io import get_model_path, load_model_data
from body_models.skel.constants import SKEL_BODY_PRESETS, SKEL_JOINTS

__all__ = ["SKEL"]


class SKEL(BodyModel):
    """SKEL body model with JAX backend."""

    NUM_BETAS = 10
    NUM_JOINTS = 24
    NUM_POSE_PARAMS = 46
    JOINTS = SKEL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ):
        if gender not in {"male", "female"}:
            raise ValueError(f"Invalid gender: {gender}. Must be 'male' or 'female'.")
        assert simplify >= 1.0

        self.gender = gender
        data = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        self.weights = common.jaxify(data)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template + self.weights.feet_offset

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def _feet_offset(self) -> Float[jax.Array, "3"]:
        return self.weights.feet_offset

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 46"],
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 46"],
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((*batch_dims, self.NUM_BETAS), dtype=dtype),
            "body_pose": jnp.zeros((*batch_dims, self.NUM_POSE_PARAMS), dtype=dtype),
            "global_rotation": jnp.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = jnp.asarray(SKEL_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = jnp.broadcast_to(body_pose, (*batch_dims, *body_pose.shape))
        return params

    def get_ipose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = jnp.asarray(SKEL_BODY_PRESETS["i_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = jnp.broadcast_to(body_pose, (*batch_dims, *body_pose.shape))
        return params
