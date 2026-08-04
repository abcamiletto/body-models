"""SMPL model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3

from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import RuntimeLike
from body_models._smpl_family import SmplFamilyModel
from body_models.smpl import _core as core
from body_models.smpl._constants import SMPL_BODY_PRESETS, SMPL_JOINT_NAMES, SMPL_JOINTS
from body_models.smpl._io import get_model_path, load_model_data

Array = Any


@dataclass(frozen=True)
class SmplConfig:
    """Static SMPL behavior preserved outside array state."""

    gender: Literal["neutral", "male", "female"]
    rotation_type: RotationType


class SMPL(SmplFamilyModel):
    """Skinned human body model with shape and pose controls."""

    NUM_JOINTS = 24
    NUM_BODY_JOINTS = 23
    NUM_SHAPE_COEFFS = 10
    _COMMON_JOINTS = SMPL_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
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
        weights = load_model_data(resolved_path, simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = SmplConfig(
            gender=gender or "neutral",
            rotation_type=rotation_type,
        )
        self._weights = runtime._materialize(weights)

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
            "pelvis_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(SMPL_JOINT_NAMES)

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
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

        pose = self.prepare_pose(body_pose, pelvis_rotation=pelvis_rotation, identity=identity)
        return self._deform_vertices(
            identity,
            pose,
            global_rotation,
            global_translation,
            vertex_indices,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
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
            body_pose,
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

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        point_regressor: PointRegressor,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is not None:
            pose = self.prepare_pose(body_pose, pelvis_rotation=pelvis_rotation, identity=identity)
            return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)
        if shape is None:
            raise ValueError("shape is required when identity is not provided")

        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
        skeleton_identity = self._prepare_skeleton_identity(shape)
        pose = self.prepare_pose(body_pose, pelvis_rotation=pelvis_rotation, identity=skeleton_identity)
        return self._deform_linear_points(
            point_regressor,
            (shape,),
            pose,
            global_rotation,
            global_translation,
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
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: core.SmplSkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            xp=self._runtime.xp,
            kinematic_fronts=self._weights.kinematic_fronts,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
    ) -> core.SmplSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            parents=self._weights.parents,
            shape=shape,
        )

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        axis_angle = self._runtime.asarray(
            SMPL_BODY_PRESETS["a_pose"],
            like=params["body_pose"],
            dtype=params["body_pose"].dtype,
        )
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            xp=self._runtime.xp,
        )
        return params


__all__ = ["SMPL", "SmplConfig"]
