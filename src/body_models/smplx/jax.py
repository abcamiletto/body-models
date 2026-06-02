"""JAX backend for SMPL-X model."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplx.backends import jax as backend
from body_models.smplx.backends.core import SmplxIdentity, SmplxPreparedPose
from body_models.smplx.io import get_model_path, load_model_data
from body_models.smplx.constants import SMPLX_BODY_PRESETS, SMPLX_HAND_PRESETS, SMPLX_JOINTS

__all__ = ["SMPLX"]


class SMPLX(BodyModel):
    """SMPL-X body model with JAX backend."""

    has_hands = True

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    NUM_JOINTS = 55
    JOINTS = SMPLX_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        """Initialize the SMPLX model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            flat_hand_mean: Whether to use a flat hand as the pose mean.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
        """
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

        resolved_path = get_model_path(model_path, gender)
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
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 55"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def exprdirs(self) -> Float[jax.Array, "V 3 E"]:
        return self.weights.exprdirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[jax.Array, "V 55"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[jax.Array, "*batch 21 N"] | Float[jax.Array, "*batch 21 3 3"],
        hand_pose: Float[jax.Array, "*batch 30 N"] | Float[jax.Array, "*batch 30 3 3"],
        head_pose: Float[jax.Array, "*batch 3 N"] | Float[jax.Array, "*batch 3 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        expression: Float[jax.Array, "*batch 10"] | None = None,
        identity: SmplxIdentity | None = None,
    ) -> Float[jax.Array, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            head_pose: Local head and facial joint rotations.
            expression: Facial expression coefficients.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            assert expression is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = jnp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression)
        pose = self.prepare_pose(body_pose, hand_pose, head_pose, pelvis_rotation, identity=identity)
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
        body_pose: Float[jax.Array, "*batch 21 N"] | Float[jax.Array, "*batch 21 3 3"],
        hand_pose: Float[jax.Array, "*batch 30 N"] | Float[jax.Array, "*batch 30 3 3"],
        head_pose: Float[jax.Array, "*batch 3 N"] | Float[jax.Array, "*batch 3 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        expression: Float[jax.Array, "*batch 10"] | None = None,
        identity: SmplxIdentity | None = None,
    ) -> Float[jax.Array, "*batch 55 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            head_pose: Local head and facial joint rotations.
            expression: Facial expression coefficients.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            assert expression is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = jnp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression=expression, skip_vertices=True)
        pose = self.prepare_pose(
            body_pose, hand_pose, head_pose, pelvis_rotation, identity=identity, skip_vertices=True
        )
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
        expression: Float[jax.Array, "*batch 10"],
        skip_vertices: bool = False,
    ) -> SmplxIdentity:
        """Precompute shape- and expression-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, expression=expression, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[jax.Array, "*batch 21 N"] | Float[jax.Array, "*batch 21 3 3"],
        hand_pose: Float[jax.Array, "*batch 30 N"] | Float[jax.Array, "*batch 30 3 3"],
        head_pose: Float[jax.Array, "*batch 3 N"] | Float[jax.Array, "*batch 3 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        expression: Float[jax.Array, "*batch 10"] | None = None,
        identity: SmplxIdentity,
        skip_vertices: bool = False,
    ) -> SmplxPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return backend.prepare_pose(
            self.weights,
            body_pose,
            hand_pose,
            head_pose,
            pelvis_rotation,
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

        body_pose_ref = jnp.zeros((*batch_dims, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        hand_pose_ref = jnp.zeros((*batch_dims, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        head_pose_ref = jnp.zeros((*batch_dims, self.NUM_HEAD_JOINTS, 3), dtype=dtype)
        pelvis_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        params = {
            "shape": jnp.zeros((*batch_dims, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(*batch_dims, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "head_pose": SO3.identity_as(
                head_pose_ref,
                batch_dims=(*batch_dims, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "expression": jnp.zeros((*batch_dims, 10), dtype=dtype),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_dims, dtype, hands)
        return params

    def _hand_preset(self, batch_dims: tuple[int, ...], dtype, hands: str):
        preset = SMPLX_HAND_PRESETS[hands]
        axis_angle = jnp.asarray(preset, dtype=dtype).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = jnp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = jnp.asarray(SMPLX_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        axis_angle = jnp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
        return params
