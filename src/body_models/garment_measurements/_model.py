"""GarmentMeasurements model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import ParameterSpec, PointRegressor, SkinnedModel, SkinningPose
from body_models._common import skinning
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import ArrayRuntime
from body_models.garment_measurements import _core as core
from body_models.garment_measurements import _pose as pose_utils
from body_models.garment_measurements._constants import (
    GARMENT_BODY_PRESETS,
    GARMENT_HAND_PRESETS,
    GARMENT_JOINTS,
)
from body_models.garment_measurements._io import get_model_path, load_model_data

Array = Any
HandPreset = Literal["default", "flat", "rest"]
GarmentMeasurementsIdentity = core.GarmentMeasurementsIdentity


@dataclass(frozen=True)
class GarmentMeasurementsConfig:
    """Static GarmentMeasurements behavior preserved outside array state."""

    rotation_type: RotationType


class GarmentMeasurements(SkinnedModel):
    """PCA body model for garment measurement workflows."""

    has_hands = True
    NUM_JOINTS = 59
    NUM_BODY_CONTROLS = 25
    NUM_HAND_CONTROLS = 30
    NUM_HEAD_CONTROLS = 3
    NUM_SHAPE_COEFFS = 15
    _COMMON_JOINTS = GARMENT_JOINTS
    _SIDE_AFFIXES = ("_L", "_R")
    _POSE_LAYOUT = pose_utils.POSE_LAYOUT

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        rotation_type: RotationType = "axis_angle",
        runtime: ArrayRuntime,
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        assets = load_model_data(get_model_path(model_path), dtype=np.float32)
        self._attach_runtime(runtime)
        self._config = GarmentMeasurementsConfig(rotation_type=rotation_type)
        self._assets = runtime._materialize(assets)

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
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_CONTROLS),
            "head_pose": ParameterSpec.rotation(rotation, count=self.NUM_HEAD_CONTROLS),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_CONTROLS),
            "pelvis_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._assets.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._assets.joint_names)

    @property
    def num_vertices(self) -> int:
        return self._assets.mean_vertices.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._assets.skin_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._assets.mean_vertices

    @property
    def parents(self) -> list[int]:
        return list(self._assets.kinematic_tree.parents)

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed GarmentMeasurements vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            identity = self.prepare_identity(*self._resolve_identity_coefficients(batch_shape, shape=shape))

        pose = self.prepare_pose(
            body_pose,
            head_pose,
            hand_pose,
            identity=identity,
            pelvis_rotation=pelvis_rotation,
        )
        vertices = self._runtime._skin_vertices(
            identity["rest_vertices"],
            pose["skinning_transforms"],
            skinning=self._assets.compact_skinning,
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
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed GarmentMeasurements joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            identity = self.prepare_identity(*self._resolve_identity_coefficients(batch_shape, shape=shape))

        packed_pose = pose_utils.pack_pose(
            xp,
            self._resolve_pelvis_rotation(body_pose, pelvis_rotation),
            body_pose,
            head_pose,
            hand_pose,
        )
        skeleton = core.prepare_skeleton(
            self._runtime,
            self._assets.bind_quats,
            self._assets.kinematic_tree,
            packed_pose,
            self.rotation_type,
            local_bind_translations=identity["local_bind_translations"],
            joint_indices=joint_indices,
        )
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            self.rotation_type,
            None,
            xp=xp,
        )

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        point_regressor: PointRegressor,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            identity = self.prepare_identity(*self._resolve_identity_coefficients(batch_shape, shape=shape))

        pose = self.prepare_pose(
            body_pose,
            head_pose,
            hand_pose,
            identity=identity,
            pelvis_rotation=pelvis_rotation,
        )
        return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

    def prepare_identity(
        self,
        shape: Float[Array, "*batch C"],
    ) -> GarmentMeasurementsIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            mean_vertices=self._assets.mean_vertices,
            components=self._assets.components,
            eigenvalues=self._assets.eigenvalues,
            bind_quats=self._assets.bind_quats,
            mvc_weights=self._assets.mvc_weights,
            kinematic_tree=self._assets.kinematic_tree,
            shape=shape,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        *,
        identity: GarmentMeasurementsIdentity,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        packed_pose = pose_utils.pack_pose(
            self._runtime.xp,
            self._resolve_pelvis_rotation(body_pose, pelvis_rotation),
            body_pose,
            head_pose,
            hand_pose,
        )
        return core.prepare_pose(
            self._runtime,
            self._assets.bind_quats,
            self._assets.kinematic_tree,
            packed_pose,
            self.rotation_type,
            bind_skeleton=identity["bind_skeleton"],
            local_bind_translations=identity["local_bind_translations"],
        )

    def _resolve_pelvis_rotation(
        self,
        body_pose: Float[Array, "*batch 25 N"] | Float[Array, "*batch 25 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    ) -> Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]:
        if pelvis_rotation is not None:
            return pelvis_rotation
        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        return SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
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
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the GarmentMeasurements T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
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
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the GarmentMeasurements rest A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)


__all__ = ["GarmentMeasurements", "GarmentMeasurementsConfig"]
