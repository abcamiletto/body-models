"""GNM Head model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float

from body_models import _pose_layout as pose_layout
from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._common.deformation import SkeletonIdentity
from body_models._linear_blendshape import LinearBlendshapeModel
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import ArrayRuntime
from body_models.gnm import _core as core
from body_models.gnm._constants import GNM_JOINTS
from body_models.gnm._io import get_model_path, load_model_data

Array = Any


@dataclass(frozen=True)
class GnmConfig:
    """Static GNM behavior preserved outside array state."""

    rotation_type: RotationType


class GNM(LinearBlendshapeModel):
    """Google's skinned statistical head model."""

    has_face = True
    NUM_JOINTS = 4
    NUM_HEAD_CONTROLS = 3
    NUM_SHAPE_COEFFS = 253
    NUM_EXPR_COEFFS = 383
    _COMMON_JOINTS = GNM_JOINTS
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

        assets = load_model_data(get_model_path(model_path), simplify=simplify)
        self._attach_runtime(runtime)
        self._config = GnmConfig(rotation_type)
        self._assets = runtime._materialize(assets)

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def joint_names(self) -> list[str]:
        return list(self._assets.joint_names)

    @property
    def identity_names(self) -> list[str]:
        """Names of GNM's identity coefficients."""
        return list(self._assets.identity_names)

    @property
    def expression_names(self) -> list[str]:
        """Names of GNM's expression coefficients."""
        return list(self._assets.expression_names)

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

    def forward_vertices(
        self,
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 253"] | None = None,
        expression: Float[Array, "*batch 383"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed GNM Head vertices."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            resolved = self._resolve_identity_coefficients(batch_shape, shape=shape, expression=expression)
            identity = self.prepare_identity(*resolved)
        pose = self.prepare_pose(head_pose, head_rotation=head_rotation, identity=identity)
        return self._deform_vertices(identity, pose, global_rotation, global_translation, vertex_indices)

    def forward_skeleton(
        self,
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 253"] | None = None,
        expression: Float[Array, "*batch 383"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch selected 4 4"]:
        """Compute posed GNM Head joint transforms."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            shape, _ = self._resolve_identity_coefficients(batch_shape, shape=shape, expression=expression)
            skeleton_identity = self._prepare_skeleton_identity(shape)
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
        return self._transform_skeleton(skeleton, global_rotation, global_translation)

    def forward_points(
        self,
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        *,
        point_regressor: PointRegressor,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 253"] | None = None,
        expression: Float[Array, "*batch 383"] | None = None,
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
        skeleton_identity = self._prepare_skeleton_identity(resolved[0])
        pose = self.prepare_pose(head_pose, head_rotation=head_rotation, identity=skeleton_identity)
        return self._deform_linear_points(point_regressor, resolved, pose, global_rotation, global_translation)

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 253"],
        expression: Float[Array, "*batch 383"],
    ) -> LinearIdentity:
        """Precompute identity- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self._assets.v_template,
            shapedirs=self._assets.shapedirs,
            exprdirs=self._assets.exprdirs,
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            parents=self._assets.kinematic_tree.parents,
            shape=shape,
            expression=expression,
        )

    def prepare_pose(
        self,
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        *,
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: SkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent skinning state."""
        return core.prepare_pose(
            self._runtime,
            self._assets.kinematic_tree,
            head_pose,
            head_rotation,
            self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(self, shape: Float[Array, "*batch 253"]) -> SkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self._assets.j_template,
            j_shapedirs=self._assets.j_shapedirs,
            parents=self._assets.kinematic_tree.parents,
            shape=shape,
        )


__all__ = ["GNM", "GnmConfig"]
