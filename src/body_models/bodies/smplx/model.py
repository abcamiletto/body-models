"""Single-source SMPL-X model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import SkinnedModel
from body_models.bodies.smplx import core
from body_models.bodies.smplx.constants import SMPLX_BODY_PRESETS, SMPLX_HAND_PRESETS, SMPLX_JOINTS
from body_models.bodies.smplx.io import get_model_path, load_model_data
from body_models.common import skinning
from body_models.rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models.runtime import ArrayRuntime
from body_models.state import StateMaterializer

Array = Any
HandPreset = Literal["default", "flat", "rest"]


@dataclass(frozen=True)
class SmplxConfig:
    """Static SMPL-X behavior preserved outside array state."""

    gender: Literal["neutral", "male", "female"]
    rotation_type: RotationType


class SMPLXModel(SkinnedModel):
    """Backend-independent SMPL-X interface and orchestration."""

    has_hands = True
    has_head = True
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    NUM_JOINTS = 55
    JOINTS = SMPLX_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        *,
        runtime: ArrayRuntime,
        materialize: StateMaterializer,
    ) -> None:
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender!r}")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        resolved_path = get_model_path(model_path, gender)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self._runtime = runtime
        self._config = SmplxConfig(gender=gender or "neutral", rotation_type=rotation_type)
        self.weights = materialize(weights)

    @property
    def gender(self) -> Literal["neutral", "male", "female"]:
        return self._config.gender

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
        return self.weights.joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V 55"]:
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
    def lbs_weights(self) -> Float[Array, "V 55"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        expression: Float[Array, "*batch 10"] | None = None,
        identity: core.SmplxIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(body_pose, hand_pose, head_pose, pelvis_rotation, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self.weights.lbs_joint_indices,
            joint_weights=self.weights.lbs_joint_weights,
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
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        expression: Float[Array, "*batch 10"] | None = None,
        identity: core.SmplxIdentity | None = None,
    ) -> Float[Array, "*batch 55 4 4"]:
        """Compute posed joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape, expression)
        else:
            skeleton_identity = identity

        skeleton = core.prepare_skeleton(
            self.weights.kinematic_fronts,
            self.weights.hand_mean,
            body_pose,
            hand_pose,
            head_pose,
            pelvis_rotation,
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
        shape: Float[Array, "*batch 10"],
        expression: Float[Array, "*batch 10"],
    ) -> core.SmplxIdentity:
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
            shape=shape,
            expression=expression,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
        hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
        head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
        pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        *,
        identity: core.SmplxIdentity,
    ) -> core.SmplxPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return core.prepare_pose(
            xp=self._runtime.xp,
            posedirs=self.weights.posedirs,
            kinematic_fronts=self.weights.kinematic_fronts,
            hand_mean=self.weights.hand_mean,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            pelvis_rotation=pelvis_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch S"],
        expression: Float[Array, "*batch E"],
    ) -> core.SmplxSkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            j_template=self.weights.j_template,
            j_shapedirs=self.weights.j_shapedirs,
            j_exprdirs=self.weights.j_exprdirs,
            parents=self.weights.parents,
            shape=shape,
            expression=expression,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero identity controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        runtime = self._runtime
        body_ref = runtime.zeros(
            (*batch_dims, self.NUM_BODY_JOINTS, 3),
            like=self.weights.v_template,
            dtype=dtype,
        )
        hand_ref = runtime.zeros(
            (*batch_dims, self.NUM_HAND_JOINTS, 3),
            like=self.weights.v_template,
            dtype=dtype,
        )
        head_ref = runtime.zeros(
            (*batch_dims, self.NUM_HEAD_JOINTS, 3),
            like=self.weights.v_template,
            dtype=dtype,
        )
        pelvis_ref = runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype)
        params = {
            "shape": runtime.zeros((*batch_dims, 10), like=self.weights.v_template, dtype=dtype),
            "expression": runtime.zeros((*batch_dims, 10), like=self.weights.v_template, dtype=dtype),
            "body_pose": SO3.identity_as(
                body_ref,
                batch_dims=(*batch_dims, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "hand_pose": SO3.identity_as(
                hand_ref,
                batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "head_pose": SO3.identity_as(
                head_ref,
                batch_dims=(*batch_dims, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=runtime.xp,
            ),
            "global_translation": runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype),
        }
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_dims, params["hand_pose"], hands)
        return params

    def _hand_preset(
        self,
        batch_dims: tuple[int, ...],
        like: Float[Array, "..."],
        hands: HandPreset,
    ) -> Float[Array, "*batch 30 N"]:
        axis_angle = self._runtime.asarray(SMPLX_HAND_PRESETS[hands], like=like).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=self._runtime.xp)

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = self._runtime.asarray(SMPLX_BODY_PRESETS["a_pose"], like=params["body_pose"])
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            xp=self._runtime.xp,
        )
        return params


__all__ = ["SMPLXModel", "SmplxConfig"]
