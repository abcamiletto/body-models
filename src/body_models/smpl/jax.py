"""JAX backend for SMPL model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smpl.backends import jax as backend
from body_models.smpl.constants import SMPL_APOSE, SMPL_IPOSE, SMPL_JOINT_NAMES, SMPL_JOINTS
from body_models.smpl.io import get_model_path, load_model_data


__all__ = ["SMPL"]


@jax.tree_util.register_pytree_node_class
class SMPL(BodyModel):
    """SMPL body model with JAX backend."""

    NUM_BODY_JOINTS = 23
    NUM_JOINTS = 24
    JOINTS = SMPL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        # Default gender to "neutral" for attribute storage when model_path is given
        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type

        resolved_path = get_model_path(model_path, gender)
        data = load_model_data(resolved_path, simplify=simplify)

        self.weights = common.jaxify(data)

    def tree_flatten(self):
        children = (self.weights,)
        aux_data = (
            self.gender,
            self.rotation_type,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.weights,) = children
        gender, rotation_type = aux_data
        obj.gender = gender
        obj.rotation_type = rotation_type
        return obj

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(SMPL_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 23 N"] | Float[jax.Array, "B 23 3 3"],
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 23 N"] | Float[jax.Array, "B 23 3 3"],
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        body_pose_ref = jnp.zeros((batch_size, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        pelvis_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        return {
            "shape": jnp.zeros((1, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(batch_size, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_size=batch_size, **kwargs)

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPL_APOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPL_IPOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params
