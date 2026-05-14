"""JAX backend for the GarmentMeasurements PCA body model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from .. import common
from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .backends import jax as backend
from .io import get_model_path, load_model_data
from .constants import GARMENT_HAND_PRESETS, GARMENT_IPOSE, GARMENT_JOINTS, GARMENT_TPOSE
from .pose import pack_pose, unpack_pose


__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    has_hands = True

    JOINTS = GARMENT_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        self.weights = common.jaxify(load_model_data(get_model_path(model_path)))
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.weights.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.mean_vertices

    @property
    def parents(self) -> list[int]:
        return [int(parent) for parent in self.weights.parents.tolist()]

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B C"],
        body_pose: Float[jax.Array, "B 25 N"] | Float[jax.Array, "B 25 3 3"],
        head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
        hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"],
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        pose = pack_pose(jnp, pelvis_rotation, body_pose, head_pose, hand_pose)
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B C"],
        body_pose: Float[jax.Array, "B 25 N"] | Float[jax.Array, "B 25 3 3"],
        head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
        hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
        pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"],
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        pose = pack_pose(jnp, pelvis_rotation, body_pose, head_pose, hand_pose)
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
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

        pose_ref = jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        global_ref = jnp.zeros((batch_size,), dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, self.num_joints),
            rotation_type=self.rotation_type,
            xp=jnp,
        )
        pelvis_rotation, body_pose, head_pose, hand_pose = unpack_pose(jnp, pose)
        if hands != "default":
            axis_angle = jnp.asarray(GARMENT_HAND_PRESETS[hands], dtype=dtype).reshape(1, hand_pose.shape[-2], 3)
            axis_angle = jnp.broadcast_to(axis_angle, hand_pose.shape)
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
        return {
            "shape": jnp.zeros((1, self.num_shape_components), dtype=dtype),
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        pose_parts = (
            params["pelvis_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        pose = pack_pose(jnp, *pose_parts)
        for joint_name, values in GARMENT_TPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            pose = common.set(pose, (slice(None), index), converted, xp=jnp)
        pelvis_rotation, body_pose, head_pose, hand_pose = unpack_pose(jnp, pose)
        params.update(
            body_pose=body_pose,
            head_pose=head_pose,
            hand_pose=hand_pose,
            pelvis_rotation=pelvis_rotation,
        )
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)

    def get_ipose(
        self,
        batch_size: int = 1,
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        pose_parts = (
            params["pelvis_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        pose = pack_pose(jnp, *pose_parts)
        for joint_name, values in GARMENT_IPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=jnp)
            pose = common.set(pose, (slice(None), index), converted, xp=jnp)
        pelvis_rotation, body_pose, head_pose, hand_pose = unpack_pose(jnp, pose)
        params.update(
            body_pose=body_pose,
            head_pose=head_pose,
            hand_pose=hand_pose,
            pelvis_rotation=pelvis_rotation,
        )
        return params
