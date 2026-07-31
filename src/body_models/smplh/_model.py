"""SMPL-H model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import LinearIdentity, ParameterSpec, SkinningPose
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import RuntimeLike
from body_models._smpl_family import SmplFamilyModel
from body_models.smplh import _core as core
from body_models.smplh._constants import SMPLH_BODY_PRESETS, SMPLH_HAND_PRESETS, SMPLH_JOINTS
from body_models.smplh._io import get_model_path, load_model_data

Array = Any
HandPreset = Literal["default", "flat", "rest"]


@dataclass(frozen=True)
class SmplhConfig:
    """Static SMPL-H behavior preserved outside array state."""

    gender: Literal["neutral", "male", "female"]
    rotation_type: RotationType


class SMPLH(SmplFamilyModel):
    """Skinned human body model with articulated hands."""

    has_hands = True
    NUM_JOINTS = 52
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_SHAPE_COEFFS = 10
    _COMMON_JOINTS = SMPLH_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender!r}")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path, gender)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = SmplhConfig(gender=gender or "neutral", rotation_type=rotation_type)
        self._weights = runtime.materialize(weights)

    @property
    def gender(self) -> Literal["neutral", "male", "female"]:
        return self._config.gender

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_JOINTS),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_JOINTS),
            "pelvis_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, hand_pose, pelvis_rotation=pelvis_rotation, identity=identity)
        return self._deform_vertices(
            identity,
            pose,
            global_rotation,
            global_translation,
            vertex_indices,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 52 4 4"]:
        """Compute posed joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self._weights.kinematic_fronts,
            self._weights.hand_mean,
            body_pose,
            hand_pose,
            pelvis_rotation,
            self.rotation_type,
            local_joint_offsets=skeleton_identity["local_joint_offsets"],
            xp=xp,
        )
        return self._transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            joint_indices,
        )

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 10"],
    ) -> LinearIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self._weights.v_template,
            shapedirs=self._weights.shapedirs,
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            parents=self._weights.parents,
            shape=shape,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: LinearIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            xp=self._runtime.xp,
            posedirs=self._weights.posedirs,
            kinematic_fronts=self._weights.kinematic_fronts,
            hand_mean=self._weights.hand_mean,
            body_pose=body_pose,
            hand_pose=hand_pose,
            pelvis_rotation=pelvis_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
    ) -> core.SmplhSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            parents=self._weights.parents,
            shape=shape,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero identity controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_dims, params["hand_pose"], hands)
        return params

    def _hand_preset(
        self,
        batch_dims: tuple[int, ...],
        like: Float[Array, "..."],
        hands: HandPreset,
    ) -> Float[Array, "*batch 30 N"]:
        axis_angle = self._runtime.asarray(SMPLH_HAND_PRESETS[hands], like=like).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=self._runtime.xp)

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-H T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-H A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        axis_angle = self._runtime.asarray(SMPLH_BODY_PRESETS["a_pose"], like=params["body_pose"])
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            xp=self._runtime.xp,
        )
        return params


__all__ = ["SMPLH", "SmplhConfig"]
