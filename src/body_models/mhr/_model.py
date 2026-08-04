"""MHR model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int

from body_models import _common as common
from body_models._base import (
    CorrectiveBasis,
    ParameterSpec,
    PointRegressor,
    SkinnedModel,
    SkinningIdentity,
    SkinningPose,
    SparseCorrectiveBasis,
)
from body_models._common import skinning
from body_models._runtime import RuntimeLike
from body_models.mhr import _core as core
from body_models.mhr._constants import (
    MHR_BODY_POSE_COEFFS,
    MHR_BODY_PRESETS,
    MHR_HAND_POSE_COEFFS,
    MHR_HAND_PRESETS,
    MHR_HEAD_POSE_COEFFS,
    MHR_JOINTS,
)
from body_models.mhr._io import get_model_path, load_model_data
from body_models.mhr._pose import pack_pose, unpack_pose

Array = Any


class MHR(SkinnedModel):
    """Expressive full-body model with neural pose correctives."""

    has_face = True
    has_hands = True
    NUM_JOINTS = 127
    NUM_SHAPE_COEFFS = 45
    NUM_EXPR_COEFFS = 72
    NUM_BODY_POSE_COEFFS = MHR_BODY_POSE_COEFFS
    NUM_HEAD_POSE_COEFFS = MHR_HEAD_POSE_COEFFS
    NUM_HAND_POSE_COEFFS = MHR_HAND_POSE_COEFFS
    NUM_POSE_COEFFS = NUM_BODY_POSE_COEFFS + NUM_HEAD_POSE_COEFFS + NUM_HAND_POSE_COEFFS
    _COMMON_JOINTS = MHR_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        lod: Literal[0, 1, 2, 3, 4, 5, 6] = 1,
        simplify: float = 1.0,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        weights = load_model_data(get_model_path(model_path), lod=lod, simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = None
        self._weights = runtime._materialize(weights)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.base_vertices.shape[0]

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "expression": ParameterSpec((self.NUM_EXPR_COEFFS,), "identity"),
            "body_pose": ParameterSpec((self.NUM_BODY_POSE_COEFFS,), "pose"),
            "head_pose": ParameterSpec((self.NUM_HEAD_POSE_COEFFS,), "pose"),
            "hand_pose": ParameterSpec((self.NUM_HAND_POSE_COEFFS,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._weights.dense_skin_weights

    @property
    def parents(self) -> list[int]:
        return list(self._weights.parents)

    @property
    def _corrective_basis(self) -> CorrectiveBasis:
        return SparseCorrectiveBasis(self._weights.correctives.basis)

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        *,
        shape: Float[Array, "*batch 45"] | None = None,
        expression: Float[Array, "*batch 72"] | None = None,
        identity: SkinningIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        vertices = self._runtime._skin_vertices(
            self.apply_pose_correctives(identity=identity, pose=pose),
            pose["skinning_transforms"],
            skinning=self._weights.compact_skinning,
            vertex_indices=vertex_indices,
        )
        return skinning.apply_global_transform(
            vertices,
            global_rotation,
            global_translation,
            xp=xp,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        *,
        shape: Float[Array, "*batch 45"] | None = None,
        expression: Float[Array, "*batch 72"] | None = None,
        identity: SkinningIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms, which are independent of identity."""
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        xp = self._runtime.xp
        pose = pack_pose(xp, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            joint_offsets=self._weights.joint_offsets,
            joint_pre_rotations=self._weights.joint_pre_rotations,
            parameter_transform=self._weights.parameter_transform,
            kinematic_fronts=self._weights.kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.NUM_SHAPE_COEFFS,
            pose=pose,
            xp=xp,
        )
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            "axis_angle",
            joint_indices,
            xp=xp,
        )

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        *,
        point_regressor: PointRegressor,
        shape: Float[Array, "*batch 45"] | None = None,
        expression: Float[Array, "*batch 72"] | None = None,
        identity: SkinningIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, expression=expression)
        if identity is None:
            if shape is None or expression is None:
                raise ValueError("shape and expression are required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 45"],
        expression: Float[Array, "*batch 72"],
    ) -> SkinningIdentity:
        """Precompute shape- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            base_vertices=self._weights.base_vertices,
            blendshape_dirs=self._weights.blendshape_dirs,
            shape=shape,
            expression=expression,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        *,
        identity: SkinningIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent MHR state."""
        pose = pack_pose(self._runtime.xp, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            joint_offsets=self._weights.joint_offsets,
            joint_pre_rotations=self._weights.joint_pre_rotations,
            parameter_transform=self._weights.parameter_transform,
            kinematic_fronts=self._weights.kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.NUM_SHAPE_COEFFS,
            bind_inv_linear=self._weights.bind_inv_linear,
            bind_inv_translation=self._weights.bind_inv_translation,
            corrective_hidden_weights=self._weights.correctives.hidden_weights,
            pose=pose,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero identity and pose controls."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            runtime = self.runtime
            hand_pose = runtime.asarray(MHR_HAND_PRESETS[hands], like=params["hand_pose"], dtype=dtype)
            params["hand_pose"] = runtime.xp.broadcast_to(
                hand_pose,
                (*batch_dims, self.NUM_HAND_POSE_COEFFS),
            )
        return params

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MHR T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        pose = self._runtime.zeros(
            (*batch_dims, self.NUM_POSE_COEFFS),
            like=params["body_pose"],
            dtype=params["body_pose"].dtype,
        )
        preset = self._runtime.asarray(
            MHR_BODY_PRESETS["t_pose"],
            like=pose,
            dtype=pose.dtype,
        )
        pose = common.at_set(pose, (..., slice(None, 100)), preset, xp=self._runtime.xp)
        params["body_pose"], params["head_pose"], _ = unpack_pose(self._runtime.xp, pose)
        return params

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MHR A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)


__all__ = ["MHR"]
