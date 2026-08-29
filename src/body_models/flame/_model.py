"""FLAME model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout
from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._linear_blendshape import LinearBlendshapeModel
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import ArrayRuntime
from body_models.flame import _core as core
from body_models.flame._constants import FLAME_JOINT_NAMES, FLAME_JOINTS
from body_models.flame._io import get_model_path, load_model_data

Array = Any


@dataclass(frozen=True)
class FlameConfig:
    """Static FLAME behavior preserved outside array state."""

    rotation_type: RotationType


class FLAME(LinearBlendshapeModel):
    """Skinned head model with shape and expression controls."""

    has_face = True
    NUM_JOINTS = 5
    NUM_HEAD_CONTROLS = 4
    NUM_SHAPE_COEFFS = 300
    NUM_EXPR_COEFFS = 100
    _COMMON_JOINTS = FLAME_JOINTS
    _SIDE_AFFIXES = ("left_", "right_")
    _POSE_LAYOUT = pose_layout.PoseLayout.per_joint(("head_rotation", 1), ("head_pose", NUM_HEAD_CONTROLS))

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: ArrayRuntime,
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path)
        assets = load_model_data(resolved_path, simplify=simplify)
        self._attach_runtime(runtime)
        self._config = FlameConfig(rotation_type=rotation_type)
        self._assets = runtime._materialize(assets)

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "expression": ParameterSpec((self.NUM_EXPR_COEFFS,), "identity"),
            "head_pose": ParameterSpec.rotation(rotation, count=self.NUM_HEAD_CONTROLS),
            "head_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    def forward_vertices(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed head vertices."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            resolved = self._resolve_identity_coefficients(batch_shape, shape=shape, expression=expression)
            identity = self.prepare_identity(*resolved)

        pose = self.prepare_pose(head_pose, head_rotation=head_rotation, identity=identity)
        return self._deform_vertices(
            identity,
            pose,
            global_rotation,
            global_translation,
            vertex_indices,
        )

    def forward_skeleton(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 5 4 4"]:
        """Compute posed head joint transforms."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            resolved = self._resolve_identity_coefficients(batch_shape, shape=shape, expression=expression)
            skeleton_identity = self._prepare_skeleton_identity(*resolved)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self._runtime,
            self._assets.kinematic_tree,
            head_pose,
            head_rotation,
            self.rotation_type,
            local_joint_offsets=skeleton_identity["local_joint_offsets"],
            joint_indices=joint_indices,
        )
        return self._transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
        )

    def forward_points(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        *,
        point_regressor: PointRegressor,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is not None:
            pose = self.prepare_pose(head_pose, head_rotation=head_rotation, identity=identity)
            return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

        batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
        resolved = self._resolve_identity_coefficients(batch_shape, shape=shape, expression=expression)
        skeleton_identity = self._prepare_skeleton_identity(*resolved)
        pose = self.prepare_pose(head_pose, head_rotation=head_rotation, identity=skeleton_identity)
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
        expression: Float[Array, "*batch E"],
    ) -> LinearIdentity:
        """Precompute shape- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self._assets.v_template,
            shapedirs=self._assets.shapedirs,
            exprdirs=self._assets.exprdirs,
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            j_exprdirs=self._assets.j_exprdirs,
            parents=self._assets.kinematic_tree.parents,
            shape=shape,
            expression=expression,
        )

    def prepare_pose(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: core.FlameSkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            self._runtime,
            self._assets.kinematic_tree,
            head_pose=head_pose,
            head_rotation=head_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
        expression: Float[Array, "*batch E"],
    ) -> core.FlameSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            j_exprdirs=self._assets.j_exprdirs,
            parents=self._assets.kinematic_tree.parents,
            shape=shape,
            expression=expression,
        )


__all__ = ["FLAME", "FlameConfig"]
