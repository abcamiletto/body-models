"""SMPL model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3

from body_models import _config as config
from body_models import _pose_layout as pose_layout
from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._common.deformation import LinearSkeletonIdentity
from body_models._common.kinematics import JointSelection
from body_models._linear_blendshape import LinearBlendshapeModel, PoseBlock
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import ArrayRuntime
from body_models.smpl._constants import SMPL_BODY_PRESETS, SMPL_JOINT_NAMES, SMPL_JOINTS
from body_models.smpl._io import load_model_data

Array = Any


@dataclass(frozen=True)
class SmplConfig:
    """Static SMPL behavior preserved outside array state."""

    gender: Literal["neutral", "male", "female"]
    rotation_type: RotationType


class SMPL(LinearBlendshapeModel):
    """Skinned human body model with shape and pose controls."""

    NUM_JOINTS = 24
    NUM_BODY_CONTROLS = 23
    NUM_SHAPE_COEFFS = 10
    _COMMON_JOINTS = SMPL_JOINTS
    _SIDE_AFFIXES = ("left_", "right_")
    _POSE_LAYOUT = pose_layout.PoseLayout.per_joint(("pelvis_rotation", 1), ("body_pose", NUM_BODY_CONTROLS))

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
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

        if (model_path is None) == (gender is None):
            raise ValueError("Pass exactly one of model_path or gender.")
        resolved_path = config.resolve_model_path("smpl", model_path, gender)
        assets = load_model_data(resolved_path, simplify=simplify)
        self._attach_runtime(runtime)
        self._config = SmplConfig(
            gender=gender or "neutral",
            rotation_type=rotation_type,
        )
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
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        identity = self._identity_from(identity, self.prepare_identity, shape=shape)
        pose = self.prepare_pose(body_pose, pelvis_rotation=pelvis_rotation, identity=identity)
        return self._deform_vertices(identity, pose, global_rotation, global_translation, vertex_indices)

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearSkeletonIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Compute posed joint transforms."""
        identity = self._identity_from(identity, self._prepare_skeleton_identity, shape=shape)
        selection = self._joint_selection(joint_indices)
        pose_matrices = self._pose_matrices(body_pose, pelvis_rotation, selection)
        skeleton = self._posed_joints(pose_matrices, identity, selection)
        return self._transform_skeleton(skeleton, global_rotation, global_translation)

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        point_regressor: PointRegressor,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        identity: LinearSkeletonIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        identity = self._identity_from(identity, self._prepare_skeleton_identity, shape=shape)
        pose = self.prepare_pose(body_pose, pelvis_rotation=pelvis_rotation, identity=identity)
        return self._deform_linear_points(point_regressor, identity, pose, global_rotation, global_translation)

    def prepare_identity(self, shape: Float[Array, "*batch S"]) -> LinearIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return self._prepare_linear_identity(shape)

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        *,
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: LinearSkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose_matrices = self._pose_matrices(body_pose, pelvis_rotation)
        return self._prepare_linear_pose(pose_matrices, identity)

    def _pose_matrices(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
        selection: JointSelection | None = None,
    ) -> Float[Array, "*batch 24 3 3"]:
        return self._assemble_pose_matrices([PoseBlock(body_pose, self.rotation_type)], pelvis_rotation, selection)

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
