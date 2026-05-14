"""JAX backend for MHR model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from body_models.mhr.backends import jax as backend
from body_models.mhr.constants import (
    MHR_BODY_POSE_DIM,
    MHR_HAND_POSE_DIM,
    MHR_IPOSE_TARGETS,
    MHR_JOINTS,
    MHR_TPOSE_TARGETS,
)
from body_models.mhr.io import get_model_path, load_model_data
from body_models.mhr.pose import pack_pose, unpack_pose

__all__ = ["MHR"]


@jax.tree_util.register_pytree_node_class
class MHR(BodyModel):
    """MHR body model with JAX backend."""

    has_hands = True

    SHAPE_DIM = 45
    EXPR_DIM = 72
    JOINTS = MHR_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        self.weights = common.jaxify(load_model_data(get_model_path(model_path), lod=lod, simplify=simplify))

    def tree_flatten(self):
        return (self.weights,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.weights,) = children
        return obj

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def body_pose_dim(self) -> int:
        return MHR_BODY_POSE_DIM

    @property
    def hand_pose_dim(self) -> int:
        return MHR_HAND_POSE_DIM

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        dense = jnp.zeros((self.weights.skin_weights.shape[0], self.num_joints), dtype=self.weights.skin_weights.dtype)
        return dense.at[jnp.arange(self.weights.skin_weights.shape[0])[:, None], self.weights.skin_indices].set(
            self.weights.skin_weights
        )

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 45"],
        body_pose: Float[jax.Array, "B 100"],
        hand_pose: Float[jax.Array, "B 104"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pack_pose(jnp, body_pose, hand_pose),
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 45"],
        body_pose: Float[jax.Array, "B 100"],
        hand_pose: Float[jax.Array, "B 104"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pack_pose(jnp, body_pose, hand_pose),
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=jnp.float32,
        hands: Literal["open", "rest"] = "rest",
    ) -> dict[str, jnp.ndarray]:
        if hands not in ("open", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'open' or 'rest'.")

        return {
            "shape": jnp.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "body_pose": jnp.zeros((batch_size, self.body_pose_dim), dtype=dtype),
            "hand_pose": jnp.zeros((batch_size, self.hand_pose_dim), dtype=dtype),
            "expression": jnp.zeros((batch_size, self.EXPR_DIM), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, jnp.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        targets = MHR_TPOSE_TARGETS
        pose = pack_pose(jnp, params["body_pose"], params["hand_pose"])
        rows = [
            next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name) * 7 + component
            for joint_name, component, _ in targets
        ]
        values = jnp.asarray([value for _, _, value in targets], dtype=pose.dtype)
        transform = jnp.asarray(self.weights.parameter_transform, dtype=pose.dtype)
        system = transform[rows, : self.pose_dim]
        pose = common.set(pose, (slice(None),), jnp.linalg.pinv(system) @ values, xp=jnp)
        params["body_pose"], params["hand_pose"] = unpack_pose(jnp, pose)
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, jnp.ndarray]:
        return self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)

    def get_ipose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, jnp.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        targets = MHR_IPOSE_TARGETS
        pose = pack_pose(jnp, params["body_pose"], params["hand_pose"])
        rows = [
            next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name) * 7 + component
            for joint_name, component, _ in targets
        ]
        values = jnp.asarray([value for _, _, value in targets], dtype=pose.dtype)
        transform = jnp.asarray(self.weights.parameter_transform, dtype=pose.dtype)
        system = transform[rows, : self.pose_dim]
        pose = common.set(pose, (slice(None),), jnp.linalg.pinv(system) @ values, xp=jnp)
        params["body_pose"], params["hand_pose"] = unpack_pose(jnp, pose)
        return params
