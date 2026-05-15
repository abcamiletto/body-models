"""JAX backend for SMPL-X model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplx.backends import jax as backend
from body_models.smplx.io import get_model_path, load_model_data
from body_models.smplx.constants import SMPLX_APOSE, SMPLX_HAND_PRESETS, SMPLX_IPOSE, SMPLX_JOINTS

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
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 21 N"] | Float[jax.Array, "B 21 3 3"],
        hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
        head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
        expression: Float[jax.Array, "B 10"] | None = None,
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 21 N"] | Float[jax.Array, "B 21 3 3"],
        hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
        head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
        expression: Float[jax.Array, "B 10"] | None = None,
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B 55 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=jnp.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, jax.Array]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        body_pose_ref = jnp.zeros((batch_size, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        hand_pose_ref = jnp.zeros((batch_size, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        head_pose_ref = jnp.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), dtype=dtype)
        pelvis_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        params = {
            "shape": jnp.zeros((1, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(batch_size, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(batch_size, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "head_pose": SO3.identity_as(
                head_pose_ref,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "expression": jnp.zeros((batch_size, 10), dtype=dtype),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_size, dtype, hands)
        return params

    def _hand_preset(self, batch_size: int, dtype, hands: str):
        preset = SMPLX_HAND_PRESETS[hands]
        axis_angle = jnp.asarray(preset, dtype=dtype).reshape(1, self.NUM_HAND_JOINTS, 3)
        axis_angle = jnp.repeat(axis_angle, batch_size, axis=0)
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)

    def get_tpose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPLX_APOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPLX_IPOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params
