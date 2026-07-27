"""MHR model implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int

from body_models import _common as common
from body_models._base import ParameterSpec, SkinnedModel
from body_models._common import skinning
from body_models._runtime import RuntimeLike
from body_models.mhr import _core as core
from body_models.mhr._constants import (
    MHR_BODY_POSE_DIM,
    MHR_BODY_PRESETS,
    MHR_HAND_POSE_DIM,
    MHR_HAND_PRESETS,
    MHR_HEAD_POSE_DIM,
    MHR_JOINTS,
)
from body_models.mhr._io import get_model_path, load_model_data
from body_models.mhr._pose import pack_pose, unpack_pose

Array = Any


class MHR(SkinnedModel):
    """Expressive full-body model with neural pose correctives."""

    has_hands = True
    has_head = True
    SHAPE_DIM = 45
    EXPR_DIM = 72
    JOINTS = MHR_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        weights = load_model_data(get_model_path(model_path), lod=lod, simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = None
        self._weights = runtime.materialize(weights)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def num_joints(self) -> int:
        return len(self._weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self._weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def body_pose_dim(self) -> int:
        return MHR_BODY_POSE_DIM

    @property
    def head_pose_dim(self) -> int:
        return MHR_HEAD_POSE_DIM

    @property
    def hand_pose_dim(self) -> int:
        return MHR_HAND_POSE_DIM

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "shape": ParameterSpec((self.SHAPE_DIM,), "identity"),
            "body_pose": ParameterSpec((self.body_pose_dim,), "pose"),
            "head_pose": ParameterSpec((self.head_pose_dim,), "pose"),
            "hand_pose": ParameterSpec((self.hand_pose_dim,), "pose"),
            "expression": ParameterSpec((self.EXPR_DIM,), "identity"),
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
        return self._weights.parents

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 45"] | None = None,
        expression: Float[Array, "*batch 72"] | None = None,
        identity: core.MhrIdentity | None = None,
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

        pose = self.prepare_pose(body_pose, head_pose, hand_pose)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self._weights.skin_indices,
            joint_weights=self._weights.skin_weights,
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
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        xp = self._runtime.xp
        pose = pack_pose(xp, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            joint_offsets=self._weights.joint_offsets,
            joint_pre_rotations=self._weights.joint_pre_rotations,
            parameter_transform=self._weights.parameter_transform,
            kinematic_fronts=self._weights.kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
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

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 45"],
        expression: Float[Array, "*batch 72"],
    ) -> core.MhrIdentity:
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
    ) -> core.MhrPreparedPose:
        """Precompute pose-dependent MHR state."""
        pose = pack_pose(self._runtime.xp, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            joint_offsets=self._weights.joint_offsets,
            joint_pre_rotations=self._weights.joint_pre_rotations,
            parameter_transform=self._weights.parameter_transform,
            kinematic_fronts=self._weights.kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            bind_inv_linear=self._weights.bind_inv_linear,
            bind_inv_translation=self._weights.bind_inv_translation,
            corrective_hidden_weights=self._weights.correctives.hidden_weights,
            corrective_output_weights=self._weights.correctives.output_weights,
            pose=pose,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero identity and pose controls."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        params = super().get_rest_pose(batch_dims, dtype)
        if hands != "default":
            runtime = self.runtime
            hand_pose = runtime.asarray(MHR_HAND_PRESETS[hands], like=params["hand_pose"], dtype=dtype)
            params["hand_pose"] = runtime.xp.broadcast_to(hand_pose, (*batch_dims, self.hand_pose_dim))
        return params

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MHR T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        pose = self._runtime.zeros(
            (*batch_dims, self.pose_dim),
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
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MHR A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


__all__ = ["MHR"]
