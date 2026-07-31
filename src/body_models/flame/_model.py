"""FLAME model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float, Int

from body_models._base import ParameterSpec
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import RuntimeLike
from body_models._smpl_family import SmplFamilyModel
from body_models.flame import _core as core
from body_models.flame._constants import FLAME_JOINT_NAMES
from body_models.flame._io import get_model_path, load_model_data

Array = Any
FlameIdentity = core.FlameIdentity
FlamePreparedPose = core.FlamePreparedPose


@dataclass(frozen=True)
class FlameConfig:
    """Static FLAME behavior preserved outside array state."""

    rotation_type: RotationType


class FLAME(SmplFamilyModel):
    """Skinned head model with shape and expression controls."""

    has_face = True
    NUM_JOINTS = 5
    NUM_HEAD_JOINTS = 4
    NUM_SHAPE_COEFFS = 300
    NUM_EXPR_COEFFS = 100

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        *,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path)
        weights = load_model_data(resolved_path, simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = FlameConfig(rotation_type=rotation_type)
        self._weights = runtime.materialize(weights)

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "expression": ParameterSpec((self.NUM_EXPR_COEFFS,), "identity"),
            "head_pose": ParameterSpec.rotation(rotation, self.NUM_HEAD_JOINTS),
            "head_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def exprdirs(self) -> Float[Array, "V 3 E"]:
        return self._weights.exprdirs

    def forward_vertices(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: FlameIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed head vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(head_pose, head_rotation, identity=identity)
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
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: FlameIdentity | None = None,
    ) -> Float[Array, "*batch 5 4 4"]:
        """Compute posed head joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = head_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape, expression)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self._weights.kinematic_fronts,
            head_pose,
            head_rotation,
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
        shape: Float[Array, "*batch S"],
        expression: Float[Array, "*batch E"],
    ) -> FlameIdentity:
        """Precompute shape- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self._weights.v_template,
            shapedirs=self._weights.shapedirs,
            exprdirs=self._weights.exprdirs,
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            j_exprdirs=self._weights.j_exprdirs,
            parents=self._weights.parents,
            shape=shape,
            expression=expression,
        )

    def prepare_pose(
        self,
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        *,
        identity: FlameIdentity,
    ) -> FlamePreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            xp=self._runtime.xp,
            posedirs=self._weights.posedirs,
            kinematic_fronts=self._weights.kinematic_fronts,
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
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            j_exprdirs=self._weights.j_exprdirs,
            parents=self._weights.parents,
            shape=shape,
            expression=expression,
        )


__all__ = ["FLAME", "FlameConfig"]
