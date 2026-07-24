"""Single-source FLAME model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import SkinnedModel
from body_models.common import skinning
from body_models.parts.flame import core
from body_models.parts.flame.constants import FLAME_JOINT_NAMES
from body_models.parts.flame.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models.runtime import ArrayRuntime
from body_models.state import StateMaterializer

Array = Any


@dataclass(frozen=True)
class FlameConfig:
    """Static FLAME behavior preserved outside array state."""

    rotation_type: RotationType


class FlameIdentityParameters(NamedTuple):
    shape: Float[Array, "*batch S"]
    expression: Float[Array, "*batch E"]


class FlameParameters(NamedTuple):
    identity: FlameIdentityParameters | core.FlameIdentity
    head_pose: Float[Array, "*batch 4 N"] | Float[Array, "*batch 4 3 3"]
    head_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]
    global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"]
    global_translation: Float[Array, "*batch 3"]


class FLAMEModel(SkinnedModel):
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
        runtime: ArrayRuntime,
        materialize: StateMaterializer,
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path)
        weights = load_model_data(resolved_path, simplify=simplify)
        self._runtime = runtime
        self._config = FlameConfig(rotation_type=rotation_type)
        self.weights = materialize(weights)

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V 5"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def exprdirs(self) -> Float[Array, "V 3 E"]:
        return self.weights.exprdirs

    @property
    def posedirs(self) -> Float[Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[Array, "V 5"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        parameters: FlameParameters,
        *,
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed head vertices."""
        xp = self._runtime.xp
        identity = self._identity(parameters)
        pose = self.prepare_pose(parameters._replace(identity=identity))
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self.weights.lbs_joint_indices,
            joint_weights=self.weights.lbs_joint_weights,
            vertex_indices=vertex_indices,
        )
        return skinning.apply_global_transform(
            vertices,
            parameters.global_rotation,
            parameters.global_translation,
            self.rotation_type,
            xp=xp,
        )

    def forward_skeleton(
        self,
        parameters: FlameParameters,
        *,
        joint_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch 5 4 4"]:
        """Compute posed head joint transforms."""
        xp = self._runtime.xp
        identity = self._identity(parameters)
        skeleton = core.prepare_skeleton(
            self.weights.kinematic_fronts,
            parameters.head_pose,
            parameters.head_rotation,
            self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            xp=xp,
        )
        return skinning.transform_skeleton(
            skeleton,
            parameters.global_rotation,
            parameters.global_translation,
            self.rotation_type,
            joint_indices,
            xp=xp,
        )

    def prepare_identity(
        self,
        parameters: FlameIdentityParameters,
    ) -> core.FlameIdentity:
        """Precompute shape- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            v_template=self.weights.v_template,
            shapedirs=self.weights.shapedirs,
            exprdirs=self.weights.exprdirs,
            j_template=self.weights.j_template,
            j_shapedirs=self.weights.j_shapedirs,
            j_exprdirs=self.weights.j_exprdirs,
            parents=self.weights.parents,
            shape=parameters.shape,
            expression=parameters.expression,
        )

    def prepare_pose(
        self,
        parameters: FlameParameters,
    ) -> core.FlamePreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        identity = self._identity(parameters)
        return core.prepare_pose(
            xp=self._runtime.xp,
            posedirs=self.weights.posedirs,
            kinematic_fronts=self.weights.kinematic_fronts,
            head_pose=parameters.head_pose,
            head_rotation=parameters.head_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> FlameParameters:
        """Return zero identity controls and identity rotations."""
        runtime = self._runtime
        head_ref = runtime.zeros(
            (*batch_dims, self.NUM_HEAD_JOINTS, 3),
            like=self.weights.v_template,
            dtype=dtype,
        )
        root_ref = runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype)
        return FlameParameters(
            identity=FlameIdentityParameters(
                shape=runtime.zeros((*batch_dims, 300), like=self.weights.v_template, dtype=dtype),
                expression=runtime.zeros((*batch_dims, 100), like=self.weights.v_template, dtype=dtype),
            ),
            head_pose=SO3.identity_as(
                head_ref,
                batch_dims=(*batch_dims, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            head_rotation=SO3.identity_as(
                root_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            global_rotation=SO3.identity_as(
                root_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            global_translation=runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype),
        )

    def prepare(self, parameters: FlameParameters) -> FlameParameters:
        return parameters._replace(identity=self._identity(parameters))

    def _identity(self, parameters: FlameParameters) -> core.FlameIdentity:
        identity = parameters.identity
        if not isinstance(identity, FlameIdentityParameters):
            return identity
        batch_shape = parameters.head_pose.shape[: -(self.num_rot_dims + 1)]
        shape = identity.shape
        expression = identity.expression
        shape = self._runtime.xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
        expression = self._runtime.xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
        return self.prepare_identity(FlameIdentityParameters(shape, expression))


__all__ = ["FLAMEModel", "FlameConfig", "FlameIdentityParameters", "FlameParameters"]
