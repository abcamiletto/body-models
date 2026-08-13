"""MANO model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3

from body_models import _pose_layout as pose_layout
from body_models._base import LinearIdentity, ParameterSpec, PointRegressor, SkinningPose
from body_models._constants import Joint
from body_models._rotations import VALID_ROTATION_TYPES, RotationType
from body_models._runtime import ArrayRuntime
from body_models._smpl_family import SmplFamilyModel
from body_models.mano import _core as core
from body_models.mano._constants import LEFT_MANO_JOINTS, MANO_HAND_PRESETS, RIGHT_MANO_JOINTS
from body_models.mano._io import get_model_path, load_model_data

Array = Any
HandPreset = Literal["default", "flat", "rest"]


@dataclass(frozen=True)
class ManoConfig:
    """Static MANO behavior preserved outside array state."""

    side: Literal["right", "left"]
    rotation_type: RotationType


class MANO(SmplFamilyModel):
    """Skinned hand model with shape and finger-pose controls."""

    has_hands = True
    NUM_JOINTS = 16
    NUM_HAND_JOINTS = 15
    NUM_SHAPE_COEFFS = 10
    _POSE_LAYOUT = pose_layout.PoseLayout.per_joint(("wrist_rotation", 1), ("hand_pose", NUM_HAND_JOINTS))

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: ArrayRuntime,
    ) -> None:
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side!r}")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path, side)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self._attach_runtime(runtime)
        self._config = ManoConfig(side=side or "right", rotation_type=rotation_type)
        self._weights = runtime._materialize(weights)

    @property
    def side(self) -> Literal["right", "left"]:
        return self._config.side

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_JOINTS),
            "wrist_rotation": ParameterSpec.rotation(rotation),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        joints = LEFT_MANO_JOINTS if self.side == "left" else RIGHT_MANO_JOINTS
        return joints

    def forward_vertices(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        *,
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed hand vertices."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = hand_pose.shape[: -(self._num_rot_dims + 1)]
            identity = self.prepare_identity(*self._resolve_identity_coefficients(batch_shape, shape=shape))

        pose = self.prepare_pose(hand_pose, wrist_rotation=wrist_rotation, identity=identity)
        return self._deform_vertices(
            identity,
            pose,
            global_rotation,
            global_translation,
            vertex_indices,
        )

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        *,
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 16 4 4"]:
        """Compute posed hand joint transforms."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            batch_shape = hand_pose.shape[: -(self._num_rot_dims + 1)]
            resolved = self._resolve_identity_coefficients(batch_shape, shape=shape)
            skeleton_identity = self._prepare_skeleton_identity(*resolved)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self._runtime,
            self._weights.kinematic_tree,
            self._weights.hand_mean,
            hand_pose,
            wrist_rotation,
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
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        *,
        point_regressor: PointRegressor,
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: LinearIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        self._validate_identity_arguments(identity, shape=shape)
        if identity is not None:
            pose = self.prepare_pose(hand_pose, wrist_rotation=wrist_rotation, identity=identity)
            return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

        batch_shape = hand_pose.shape[: -(self._num_rot_dims + 1)]
        resolved = self._resolve_identity_coefficients(batch_shape, shape=shape)
        skeleton_identity = self._prepare_skeleton_identity(*resolved)
        pose = self.prepare_pose(hand_pose, wrist_rotation=wrist_rotation, identity=skeleton_identity)
        return self._deform_linear_points(
            point_regressor,
            resolved,
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
            parents=self._weights.kinematic_tree.parents,
            shape=shape,
        )

    def prepare_pose(
        self,
        hand_pose: Float[Array, "*batch 15 N"] | Float[Array, "*batch 15 3 3"],
        *,
        wrist_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        identity: core.ManoSkeletonIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            self._runtime,
            self._weights.kinematic_tree,
            hand_mean=self._weights.hand_mean,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
    ) -> core.ManoSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self._weights.j_template,
            j_shapedirs=self._weights.j_shapedirs,
            parents=self._weights.kinematic_tree.parents,
            shape=shape,
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
            params["hand_pose"] = self._hand_preset(batch_dims, params["hand_pose"], hands)
        return params

    def _hand_preset(
        self,
        batch_dims: tuple[int, ...],
        like: Float[Array, "..."],
        hands: HandPreset,
    ) -> Float[Array, "*batch 15 N"]:
        axis_angle = self._runtime.asarray(MANO_HAND_PRESETS[self.side][hands], like=like).reshape(
            self.NUM_HAND_JOINTS,
            3,
        )
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=self._runtime.xp)


__all__ = ["MANO", "ManoConfig"]
