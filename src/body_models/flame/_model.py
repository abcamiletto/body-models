"""Single-source FLAME model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float, Int

from body_models._base import ParameterSpec, SkinnedModel
from body_models._common import skinning
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import RuntimeLike
from body_models.flame import _core as core
from body_models.flame._constants import FLAME_JOINT_NAMES
from body_models.flame._io import get_model_path, load_model_data

Array = Any


@dataclass(frozen=True)
class FlameConfig:
    """Static FLAME behavior preserved outside array state."""

    rotation_type: RotationType


class FLAME(SkinnedModel):
    """Backend-independent FLAME interface and orchestration."""

    has_head = True
    NUM_HEAD_JOINTS = 4
    NUM_JOINTS = 5

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
    def num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((300,), "identity"),
            "expression": ParameterSpec((100,), "identity"),
            "head_pose": ParameterSpec.rotation(rotation, self.NUM_HEAD_JOINTS),
            "head_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self._weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V 5"]:
        return self._weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.v_template

    @property
    def shapedirs(self) -> Float[Array, "V 3 S"]:
        return self._weights.shapedirs

    @property
    def exprdirs(self) -> Float[Array, "V 3 E"]:
        return self._weights.exprdirs

    @property
    def posedirs(self) -> Float[Array, "P V*3"]:
        return self._weights.posedirs

    @property
    def lbs_weights(self) -> Float[Array, "V 5"]:
        return self._weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self._weights.parents

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
        identity: core.FlameIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed head vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = head_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(head_pose, head_rotation, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self._weights.lbs_joint_indices,
            joint_weights=self._weights.lbs_joint_weights,
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
        head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"],
        head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch S"] | None = None,
        expression: Float[Array, "*batch E"] | None = None,
        identity: core.FlameIdentity | None = None,
    ) -> Float[Array, "*batch 5 4 4"]:
        """Compute posed head joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = head_pose.shape[: -(self.num_rot_dims + 1)]
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
        shape: Float[Array, "*batch S"],
        expression: Float[Array, "*batch E"],
    ) -> core.FlameIdentity:
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
        identity: core.FlameIdentity,
    ) -> core.FlamePreparedPose:
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
