"""JAX backend for MANO model."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.mano.backends import jax as backend
from body_models.mano.backends.core import ManoIdentity, ManoPreparedPose
from body_models.mano.io import get_model_path, load_model_data
from body_models.mano.constants import LEFT_MANO_JOINTS, MANO_HAND_PRESETS, RIGHT_MANO_JOINTS
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["MANO"]


class MANO(BodyModel):
    """MANO hand model with JAX backend."""

    has_hands = True

    NUM_HAND_JOINTS = 15
    NUM_JOINTS = 16

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        """Initialize the MANO model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            side: Hand side to load.
            flat_hand_mean: Whether to use a flat hand as the pose mean.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
        """
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side}. Must be 'right' or 'left'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        self.side = side if side is not None else "right"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

        resolved_path = get_model_path(model_path, side)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self.weights = common.jaxify(weights)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def common_joints(self):
        return LEFT_MANO_JOINTS if self.side == "left" else RIGHT_MANO_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 16"]:
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
    def lbs_weights(self) -> Float[jax.Array, "V 16"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        hand_pose: Float[jax.Array, "B 15 N"] | Float[jax.Array, "B 15 3 3"],
        wrist_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: ManoIdentity | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            hand_pose: Local hand joint rotations.
            wrist_rotation: Root wrist rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.
            shape: Shape coefficients.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = tuple(hand_pose.shape[: -(self.num_rot_dims + 1)])
            identity = self.prepare_identity(jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1])))
        pose = self.prepare_pose(hand_pose, wrist_rotation, identity=identity)
        return backend.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=pose["skinning_transforms"],
            pose_offsets=pose["pose_offsets"],
        )

    def forward_skeleton(
        self,
        hand_pose: Float[jax.Array, "B 15 N"] | Float[jax.Array, "B 15 3 3"],
        wrist_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: ManoIdentity | None = None,
    ) -> Float[jax.Array, "B 16 4 4"]:
        """Compute posed joint transforms.

        Args:
            hand_pose: Local hand joint rotations.
            wrist_rotation: Root wrist rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.
            shape: Shape coefficients.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = tuple(hand_pose.shape[: -(self.num_rot_dims + 1)])
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        pose = self.prepare_pose(hand_pose, wrist_rotation, identity=identity, skip_vertices=True)
        return backend.forward_skeleton(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            skeleton_transforms=pose["skeleton_transforms"],
        )

    def prepare_identity(
        self,
        shape: Float[jax.Array, "*batch 10"],
        skip_vertices: bool = False,
    ) -> ManoIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        hand_pose: Float[jax.Array, "B 15 N"] | Float[jax.Array, "B 15 3 3"],
        wrist_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        *,
        identity: ManoIdentity,
        skip_vertices: bool = False,
    ) -> ManoPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return backend.prepare_pose(
            self.weights,
            hand_pose,
            wrist_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=jnp.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, jax.Array]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        hand_pose_ref = jnp.zeros((*batch_dims, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        wrist_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        hand_pose = SO3.identity_as(
            hand_pose_ref,
            batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
            rotation_type=self.rotation_type,
            xp=jnp,
        )
        if hands != "default":
            hand_pose = self._hand_preset(batch_dims, dtype, hands)
        return {
            "shape": jnp.zeros((*batch_dims, 10), dtype=dtype),
            "hand_pose": hand_pose,
            "wrist_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def _hand_preset(self, batch_dims: tuple[int, ...], dtype, hands: str):
        preset = MANO_HAND_PRESETS[self.side][hands]
        axis_angle = jnp.asarray(preset, dtype=dtype).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = jnp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
