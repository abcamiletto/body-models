"""GarmentMeasurements model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import ParameterSpec, SkinnedModel
from body_models._common import skinning
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import RuntimeLike
from body_models.garment_measurements import _core as core
from body_models.garment_measurements._constants import (
    GARMENT_BODY_PRESETS,
    GARMENT_HAND_PRESETS,
    GARMENT_JOINTS,
)
from body_models.garment_measurements._io import get_model_path, load_model_data
from body_models.garment_measurements._pose import pack_pose

Array = Any
HandPreset = Literal["default", "flat", "rest"]
GarmentMeasurementsIdentity = core.GarmentMeasurementsIdentity
GarmentMeasurementsPreparedPose = core.GarmentMeasurementsPreparedPose


@dataclass(frozen=True)
class GarmentMeasurementsConfig:
    """Static GarmentMeasurements behavior preserved outside array state."""

    rotation_type: RotationType


class GarmentMeasurements(SkinnedModel):
    """PCA body model for garment measurement workflows."""

    has_hands = True
    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    _COMMON_JOINTS = GARMENT_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        rotation_type: RotationType = "axis_angle",
        runtime: RuntimeLike = "numpy",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        weights = load_model_data(get_model_path(model_path), dtype=np.float32)
        runtime = self._set_runtime(runtime)
        self._config = GarmentMeasurementsConfig(rotation_type=rotation_type)
        self._weights = runtime.materialize(weights)

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def _num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.num_shape_components,), "identity"),
            "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_JOINTS),
            "head_pose": ParameterSpec.rotation(rotation, count=self.NUM_HEAD_JOINTS),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_JOINTS),
            "pelvis_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self._weights.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.mean_vertices

    @property
    def parents(self) -> list[int]:
        return [int(parent) for parent in self._weights.parents.tolist()]

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"],
        *,
        shape: Float[Array, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed GarmentMeasurements vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, pelvis_rotation, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"],
            pose["skinning_transforms"],
            skinning=self._weights.compact_skinning,
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
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"],
        *,
        shape: Float[Array, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed GarmentMeasurements joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        packed_pose = pack_pose(
            xp,
            pelvis_rotation,
            body_pose,
            head_pose,
            hand_pose,
        )
        skeleton = core.prepare_skeleton(
            self._weights.bind_quats,
            self._weights.kinematic_fronts,
            packed_pose,
            self.rotation_type,
            local_bind_translations=identity["local_bind_translations"],
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
        shape: Float[Array, "*batch C"],
    ) -> GarmentMeasurementsIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            mean_vertices=self._weights.mean_vertices,
            components=self._weights.components,
            eigenvalues=self._weights.eigenvalues,
            bind_quats=self._weights.bind_quats,
            mvc_weights=self._weights.mvc_weights,
            kinematic_fronts=self._weights.kinematic_fronts,
            shape=shape,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"],
        *,
        identity: GarmentMeasurementsIdentity,
    ) -> GarmentMeasurementsPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        packed_pose = pack_pose(
            self._runtime.xp,
            pelvis_rotation,
            body_pose,
            head_pose,
            hand_pose,
        )
        return core.prepare_pose(
            self._weights.bind_quats,
            self._weights.kinematic_fronts,
            packed_pose,
            self.rotation_type,
            bind_skeleton=identity["bind_skeleton"],
            local_bind_translations=identity["local_bind_translations"],
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero shape controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            runtime = self.runtime
            axis_angle = runtime.asarray(GARMENT_HAND_PRESETS[hands], like=params["hand_pose"]).reshape(-1, 3)
            axis_angle = runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            params["hand_pose"] = SO3.convert(
                axis_angle,
                src="axis_angle",
                dst=self.rotation_type,
                xp=runtime.xp,
            )
        return params

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the GarmentMeasurements T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = self._runtime.asarray(GARMENT_BODY_PRESETS["t_pose"], like=params["body_pose"])
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            xp=self._runtime.xp,
        )
        return params

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the GarmentMeasurements rest A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


__all__ = ["GarmentMeasurements", "GarmentMeasurementsConfig"]
