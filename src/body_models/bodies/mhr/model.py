"""Single-source MHR model program."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int

from body_models import common
from body_models.base import SkinnedModel
from body_models.bodies.mhr import core
from body_models.bodies.mhr.constants import (
    MHR_BODY_POSE_DIM,
    MHR_BODY_PRESETS,
    MHR_HAND_POSE_DIM,
    MHR_HAND_PRESETS,
    MHR_HEAD_POSE_DIM,
    MHR_JOINTS,
)
from body_models.bodies.mhr.io import get_model_path, load_model_data
from body_models.bodies.mhr.pose import pack_pose, unpack_pose
from body_models.common import skinning
from body_models.runtime import Runtime

Array = Any


class MHRModel(SkinnedModel):
    """Backend-independent MHR interface and orchestration."""

    identity_keys = ("shape", "expression")
    pose_keys = ("body_pose", "head_pose", "hand_pose")
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
        runtime: Runtime,
    ) -> None:
        weights = load_model_data(get_model_path(model_path), lod=lod, simplify=simplify)
        self._runtime = runtime
        self._config = None
        self.weights = runtime.convert_model_data(weights)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

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
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._runtime.expand_skinning_weights(
            self.weights.skin_indices,
            self.weights.skin_weights,
            self.num_joints,
        )

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        expression: Float[Array, "*batch 72"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 45"] | None = None,
        identity: core.MhrIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self.weights.skin_indices,
            joint_weights=self.weights.skin_weights,
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
        expression: Float[Array, "*batch 72"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 45"] | None = None,
        identity: core.MhrIdentity | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            expression = xp.broadcast_to(expression, (*batch_shape, expression.shape[-1]))
            identity = self.prepare_identity(shape, expression, skip_vertices=True)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, skip_vertices=True)
        return skinning.transform_skeleton(
            pose["skeleton_transforms"],
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
        skip_vertices: bool = False,
    ) -> core.MhrIdentity:
        """Precompute shape- and expression-dependent state."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            base_vertices=None if skip_vertices else self.weights.base_vertices,
            blendshape_dirs=None if skip_vertices else self.weights.blendshape_dirs,
            shape=shape,
            expression=expression,
            skip_vertices=skip_vertices,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 94"],
        head_pose: Float[Array, "*batch 6"],
        hand_pose: Float[Array, "*batch 104"],
        *,
        skip_vertices: bool = False,
    ) -> core.MhrPreparedPose:
        """Precompute pose-dependent MHR state."""
        pose = pack_pose(self._runtime.xp, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            joint_offsets=self.weights.joint_offsets,
            joint_pre_rotations=self.weights.joint_pre_rotations,
            parameter_transform=self.weights.parameter_transform,
            kinematic_fronts=self.weights.kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            bind_inv_linear=self.weights.bind_inv_linear,
            bind_inv_translation=self.weights.bind_inv_translation,
            corrective_W1=self.weights.corrective_W1,
            corrective_W2=self.weights.corrective_W2,
            pose=pose,
            skip_vertices=skip_vertices,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Array]:
        """Return zero identity and pose controls."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        runtime = self._runtime
        reference = self.weights.base_vertices
        hand_pose = runtime.zeros((*batch_dims, self.hand_pose_dim), like=reference, dtype=dtype)
        if hands != "default":
            hand_pose = runtime.asarray(MHR_HAND_PRESETS[hands], like=reference, dtype=dtype)
            hand_pose = runtime.xp.broadcast_to(hand_pose, (*batch_dims, self.hand_pose_dim))
        return {
            "shape": runtime.zeros((*batch_dims, self.SHAPE_DIM), like=reference, dtype=dtype),
            "body_pose": runtime.zeros((*batch_dims, self.body_pose_dim), like=reference, dtype=dtype),
            "head_pose": runtime.zeros((*batch_dims, self.head_pose_dim), like=reference, dtype=dtype),
            "hand_pose": hand_pose,
            "expression": runtime.zeros((*batch_dims, self.EXPR_DIM), like=reference, dtype=dtype),
            "global_rotation": runtime.zeros((*batch_dims, 3), like=reference, dtype=dtype),
            "global_translation": runtime.zeros((*batch_dims, 3), like=reference, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
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
        pose = common.set(pose, (..., slice(None, 100)), preset, xp=self._runtime.xp)
        params["body_pose"], params["head_pose"], _ = unpack_pose(self._runtime.xp, pose)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return the MHR A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


__all__ = ["MHRModel"]
