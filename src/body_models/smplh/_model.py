"""SMPL-H model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3

from body_models import _pose_layout as pose_layout
from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import ArrayRuntime
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
    NUM_BODY_CONTROLS = 21
    NUM_HAND_CONTROLS = 30
    NUM_SHAPE_COEFFS = 16
    _COMMON_JOINTS = SMPLH_JOINTS
    _POSE_LAYOUT = pose_layout.PoseLayout.per_joint(
        ("pelvis_rotation", 1),
        ("body_pose", NUM_BODY_CONTROLS),
        ("hand_pose", NUM_HAND_CONTROLS),
    )

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: ArrayRuntime,
    ) -> None:
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender!r}")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path, gender)
        assets = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self._attach_runtime(runtime)
        self._config = SmplhConfig(gender=gender or "neutral", rotation_type=rotation_type)
        self._assets = runtime._materialize(assets)

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
            "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_CONTROLS),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_CONTROLS),
            "pelvis_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(self._assets.joint_names)

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            identity = self.prepare_identity(*self._resolve_identity_coefficients(batch_shape, shape=shape))

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
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 52 4 4"]:
        """Compute posed joint transforms."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            resolved = self._resolve_identity_coefficients(batch_shape, shape=shape)
            skeleton_identity = self._prepare_skeleton_identity(*resolved)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self._runtime,
            self._assets.kinematic_tree,
            self._assets.hand_mean,
            body_pose,
            hand_pose,
            pelvis_rotation,
            self.rotation_type,
            local_joint_offsets=skeleton_identity["local_joint_offsets"],
        )
        return self._transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            joint_indices,
        )

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        point_regressor: PointRegressor,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is not None:
            pose = self.prepare_pose(
                body_pose,
                hand_pose,
                pelvis_rotation=pelvis_rotation,
                identity=identity,
            )
            return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        resolved = self._resolve_identity_coefficients(batch_shape, shape=shape)
        skeleton_identity = self._prepare_skeleton_identity(*resolved)
        pose = self.prepare_pose(
            body_pose,
            hand_pose,
            pelvis_rotation=pelvis_rotation,
            identity=skeleton_identity,
        )
        return self._deform_linear_points(
            point_regressor,
            resolved,
            pose,
            global_rotation,
            global_translation,
        )

    def prepare_identity(
        self,
        shape: Float[Array, "*batch S"],
    ) -> LinearIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self._assets.v_template,
            shapedirs=self._assets.shapedirs,
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            parents=self._assets.kinematic_tree.parents,
            shape=shape,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: core.SmplhSkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            self._runtime,
            self._assets.kinematic_tree,
            hand_mean=self._assets.hand_mean,
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
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            parents=self._assets.kinematic_tree.parents,
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
        axis_angle = self._runtime.asarray(SMPLH_HAND_PRESETS[hands], like=like).reshape(self.NUM_HAND_CONTROLS, 3)
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
