"""SKEL model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int

from body_models._base import ParameterSpec, SkinnedModel
from body_models._common import skinning
from body_models._runtime import RuntimeLike
from body_models.skel import _core as core
from body_models.skel._constants import SKEL_BODY_PRESETS, SKEL_JOINTS
from body_models.skel._io import get_model_path, load_model_data
from body_models.skel._pose import (
    SKEL_BODY_POSE_DIM,
    SKEL_CANONICAL_POSE_DIM,
    SKEL_HEAD_POSE_DIM,
    pack_pose,
    unpack_pose,
)

Array = Any


@dataclass(frozen=True)
class SkelConfig:
    """Static SKEL behavior preserved outside array state."""

    gender: Literal["male", "female"]


class SKEL(SkinnedModel):
    """Skinned body model with anatomical articulation."""

    NUM_BETAS = 10
    NUM_JOINTS = 24
    JOINTS = SKEL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
        *,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        if gender not in ("male", "female"):
            raise ValueError(f"Invalid gender: {gender!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        weights = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        runtime = self._set_runtime(runtime)
        self._config = SkelConfig(gender=gender)
        self._weights = runtime.materialize(weights)

    @property
    def gender(self) -> Literal["male", "female"]:
        return self._config.gender

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def skeleton_faces(self) -> Int[Array, "Fs 3"]:
        return self._weights.skel_faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V 24"]:
        return self._weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.v_template

    @property
    def shapedirs(self) -> Float[Array, "V 3 B"]:
        return self._weights.shapedirs

    @property
    def posedirs(self) -> Float[Array, "P V*3"]:
        return self._weights.posedirs

    @property
    def parents(self) -> list[int]:
        return list(self._weights.parents)

    @property
    def pose_dim(self) -> int:
        return SKEL_CANONICAL_POSE_DIM

    @property
    def body_pose_dim(self) -> int:
        return SKEL_BODY_POSE_DIM

    @property
    def head_pose_dim(self) -> int:
        return SKEL_HEAD_POSE_DIM

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "shape": ParameterSpec((self.NUM_BETAS,), "identity"),
            "body_pose": ParameterSpec((self.body_pose_dim,), "pose"),
            "head_pose": ParameterSpec((self.head_pose_dim,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 43"],
        head_pose: Float[Array, "*batch 3"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.SkelIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed SKEL vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, head_pose, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
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
        body_pose: Float[Array, "*batch 43"],
        head_pose: Float[Array, "*batch 3"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Int[Array, "S"] | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.SkelIdentity | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Compute posed SKEL joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[:-1]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape)
        else:
            skeleton_identity = identity

        packed_pose = pack_pose(xp, body_pose, head_pose)
        skeleton = core.prepare_skeleton(
            all_axes=self._weights.all_axes,
            rotation_indices=self._weights.rotation_indices,
            apose_R=self._weights.apose_R,
            apose_t=self._weights.apose_t,
            per_joint_rot=self._weights.per_joint_rot,
            child=self._weights.child,
            fixed_orientation_joints=self._weights.fixed_orientation_joints,
            scapula_r_axes=self._weights.scapula_r_axes,
            scapula_l_axes=self._weights.scapula_l_axes,
            spine_axes=self._weights.spine_axes,
            parents=self._weights.parents,
            pose=packed_pose,
            local_joint_offsets=skeleton_identity["local_joint_offsets"],
            rest_joints=skeleton_identity["rest_joints"],
            xp=xp,
        )
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            joint_indices=joint_indices,
            xp=xp,
        )

    def forward_links(
        self,
        body_pose: Float[Array, "*batch 43"],
        head_pose: Float[Array, "*batch 3"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.SkelIdentity | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Alias the SKEL joint transforms as anatomical link transforms."""
        return self.forward_skeleton(
            body_pose,
            head_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            shape=shape,
            identity=identity,
        )

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 10"],
    ) -> core.SkelIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return core.prepare_identity(
            self._weights.v_template,
            self._weights.shapedirs,
            self._weights.j_template,
            self._weights.j_shapedirs,
            self._weights.parent,
            shape,
            xp=self._runtime.xp,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 43"],
        head_pose: Float[Array, "*batch 3"],
        *,
        identity: core.SkelIdentity,
    ) -> core.SkelPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        packed_pose = pack_pose(self._runtime.xp, body_pose, head_pose)
        return core.prepare_pose(
            all_axes=self._weights.all_axes,
            rotation_indices=self._weights.rotation_indices,
            apose_R=self._weights.apose_R,
            apose_t=self._weights.apose_t,
            per_joint_rot=self._weights.per_joint_rot,
            child=self._weights.child,
            fixed_orientation_joints=self._weights.fixed_orientation_joints,
            scapula_r_axes=self._weights.scapula_r_axes,
            scapula_l_axes=self._weights.scapula_l_axes,
            spine_axes=self._weights.spine_axes,
            parents=self._weights.parents,
            num_joints_smpl=self._weights.num_joints_smpl,
            posedirs=self._weights.posedirs,
            pose=packed_pose,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
            xp=self._runtime.xp,
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch 10"],
    ) -> core.SkelSkeletonIdentity:
        return core.prepare_skeleton_identity(
            self._weights.j_template,
            self._weights.j_shapedirs,
            self._weights.parent,
            shape,
            xp=self._runtime.xp,
        )

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Float[Array, "..."]]:
        """Return the SKEL T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Float[Array, "..."]]:
        """Return the SKEL A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        pose = self._runtime.asarray(SKEL_BODY_PRESETS["a_pose"], like=params["body_pose"])
        pose = self._runtime.xp.broadcast_to(pose, (*batch_dims, *pose.shape))
        params["body_pose"], params["head_pose"] = unpack_pose(self._runtime.xp, pose)
        return params


__all__ = ["SKEL", "SkelConfig"]
