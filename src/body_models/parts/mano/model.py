"""Single-source MANO model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import SkinnedModel
from body_models.common import skinning
from body_models.parts.mano import core
from body_models.parts.mano.constants import LEFT_MANO_JOINTS, MANO_HAND_PRESETS, RIGHT_MANO_JOINTS
from body_models.parts.mano.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models.runtime import ArrayRuntime
from body_models.state import StateMaterializer

Array = Any
HandPreset = Literal["default", "flat", "rest"]


@dataclass(frozen=True)
class ManoConfig:
    """Static MANO behavior preserved outside array state."""

    side: Literal["right", "left"]
    rotation_type: RotationType


class MANOModel(SkinnedModel):
    """Backend-independent MANO interface and orchestration."""

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
        *,
        runtime: ArrayRuntime,
        materialize: StateMaterializer,
    ) -> None:
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side!r}")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path, side)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self._runtime = runtime
        self._config = ManoConfig(side=side or "right", rotation_type=rotation_type)
        self.weights = materialize(weights)

    @property
    def side(self) -> Literal["right", "left"]:
        return self._config.side

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def faces(self) -> Int[Array, "F 3"]:
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
    def skin_weights(self) -> Float[Array, "V 16"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[Array, "V 16"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.ManoIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed hand vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = hand_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(hand_pose, wrist_rotation, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self.weights.lbs_joint_indices,
            joint_weights=self.weights.lbs_joint_weights,
            vertex_indices=vertex_indices,
        )
        return skinning.apply_global_transform(
            vertices,
            global_rotation,
            global_translation,
            self.rotation_type,
            xp=xp,
        )

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.ManoIdentity | None = None,
    ) -> Float[Array, "*batch 16 4 4"]:
        """Compute posed hand joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = hand_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self.weights.kinematic_fronts,
            self.weights.hand_mean,
            hand_pose,
            wrist_rotation,
            self.rotation_type,
            local_joint_offsets=skeleton_identity["local_joint_offsets"],
            xp=xp,
        )
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            self.rotation_type,
            joint_indices,
            xp=xp,
        )

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 10"],
    ) -> core.ManoIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self.weights.v_template,
            shapedirs=self.weights.shapedirs,
            j_template=self.weights.j_template,
            j_shapedirs=self.weights.j_shapedirs,
            parents=self.weights.parents,
            shape=shape,
        )

    def prepare_pose(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        *,
        identity: core.ManoIdentity,
    ) -> core.ManoPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            xp=self._runtime.xp,
            posedirs=self.weights.posedirs,
            kinematic_fronts=self.weights.kinematic_fronts,
            hand_mean=self.weights.hand_mean,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
    ) -> core.ManoSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self.weights.j_template,
            j_shapedirs=self.weights.j_shapedirs,
            parents=self.weights.parents,
            shape=shape,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero shape controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        runtime = self._runtime
        hand_ref = runtime.zeros(
            (*batch_dims, self.NUM_HAND_JOINTS, 3),
            like=self.weights.v_template,
            dtype=dtype,
        )
        wrist_ref = runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype)
        hand_pose = SO3.identity_as(
            hand_ref,
            batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
            rotation_type=self.rotation_type,
            xp=runtime.xp,
        )
        if hands != "default":
            hand_pose = self._hand_preset(batch_dims, hand_pose, hands)
        return {
            "shape": runtime.zeros((*batch_dims, 10), like=self.weights.v_template, dtype=dtype),
            "hand_pose": hand_pose,
            "wrist_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "global_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "global_translation": runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype),
        }

    def _hand_preset(
        self,
        batch_dims: tuple[int, ...],
        like: Float[Array, "..."],
        hands: HandPreset,
    ) -> Float[Array, "*batch 15 N"]:
        axis_angle = self._runtime.asarray(MANO_HAND_PRESETS[self.side][hands], like=like).reshape(
            self.NUM_HAND_JOINTS,
            3,
        )
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=self._runtime.xp)


__all__ = ["MANOModel", "ManoConfig"]
