"""JAX backend for MHR model."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import SkinnedModel
from body_models.bodies.mhr.backends import jax as backend
from body_models.bodies.mhr.backends.core import MhrIdentity, MhrPreparedPose
from body_models.bodies.mhr.constants import (
    MHR_BODY_POSE_DIM,
    MHR_HEAD_POSE_DIM,
    MHR_HAND_PRESETS,
    MHR_HAND_POSE_DIM,
    MHR_BODY_PRESETS,
    MHR_JOINTS,
)
from body_models.bodies.mhr.io import get_model_path, load_model_data
from body_models.bodies.mhr.pose import pack_pose, unpack_pose

__all__ = ["MHR"]


@jax.tree_util.register_pytree_node_class
class MHR(SkinnedModel):
    """MHR body model with JAX backend."""

    has_hands = True
    has_head = True
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
        """Initialize the MHR model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            lod: Level-of-detail variant to load.
            simplify: Mesh simplification factor to apply while loading.
        """
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
    def head_pose_dim(self) -> int:
        return MHR_HEAD_POSE_DIM

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
        body_pose: Float[jax.Array, "*batch 94"],
        head_pose: Float[jax.Array, "*batch 6"],
        hand_pose: Float[jax.Array, "*batch 104"],
        expression: Float[jax.Array, "*batch 72"],
        global_rotation: Float[jax.Array, "*batch 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 45"] | None = None,
        identity: MhrIdentity | None = None,
    ) -> Float[jax.Array, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial controls.
            hand_pose: Local hand joint rotations.
            expression: Facial expression coefficients.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = jnp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose)
        return backend.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=pose["skinning_transforms"],
            pose_offsets=pose["pose_offsets"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "*batch 94"],
        head_pose: Float[jax.Array, "*batch 6"],
        hand_pose: Float[jax.Array, "*batch 104"],
        expression: Float[jax.Array, "*batch 72"],
        global_rotation: Float[jax.Array, "*batch 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 45"] | None = None,
        identity: MhrIdentity | None = None,
    ) -> Float[jax.Array, "*batch J 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial controls.
            hand_pose: Local hand joint rotations.
            expression: Facial expression coefficients.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = jnp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression, skip_vertices=True)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, skip_vertices=True)
        return backend.forward_skeleton(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            skeleton_transforms=pose["skeleton_transforms"],
        )

    def prepare_identity(
        self,
        shape: Float[jax.Array, "*batch 45"],
        expression: Float[jax.Array, "*batch 72"],
        skip_vertices: bool = False,
    ) -> MhrIdentity:
        """Precompute shape- and expression-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, expression=expression, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[jax.Array, "*batch 94"],
        head_pose: Float[jax.Array, "*batch 6"],
        hand_pose: Float[jax.Array, "*batch 104"],
        *,
        identity: MhrIdentity | None = None,
        skip_vertices: bool = False,
    ) -> MhrPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose = pack_pose(jnp, body_pose, head_pose, hand_pose)
        return backend.prepare_pose(self.weights, pose, skip_vertices=skip_vertices)

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=jnp.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, jnp.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        hand_pose = jnp.zeros((*batch_dims, self.hand_pose_dim), dtype=dtype)
        if hands != "default":
            hand_pose = jnp.asarray(MHR_HAND_PRESETS[hands], dtype=dtype).reshape(self.hand_pose_dim)
            hand_pose = jnp.broadcast_to(hand_pose, (*batch_dims, self.hand_pose_dim))
        return {
            "shape": jnp.zeros((*batch_dims, self.SHAPE_DIM), dtype=dtype),
            "body_pose": jnp.zeros((*batch_dims, self.body_pose_dim), dtype=dtype),
            "head_pose": jnp.zeros((*batch_dims, self.head_pose_dim), dtype=dtype),
            "hand_pose": hand_pose,
            "expression": jnp.zeros((*batch_dims, self.EXPR_DIM), dtype=dtype),
            "global_rotation": jnp.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jnp.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        pose = jnp.zeros((*batch_dims, self.pose_dim), dtype=params["body_pose"].dtype)
        pose = pose.at[..., :100].set(jnp.asarray(MHR_BODY_PRESETS["t_pose"], dtype=pose.dtype))
        params["body_pose"], params["head_pose"], _hand_pose = unpack_pose(jnp, pose)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jnp.ndarray]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
