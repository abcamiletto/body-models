"""Single-source SKEL model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int

from body_models.base import SkinnedModel
from body_models.common import skinning
from body_models.runtime import Runtime
from body_models.skeletons.skel import core
from body_models.skeletons.skel.constants import SKEL_BODY_PRESETS, SKEL_JOINTS
from body_models.skeletons.skel.io import get_model_path, load_model_data
from body_models.skeletons.skel.pose import (
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


class SKELModel(SkinnedModel):
    """Backend-independent SKEL interface and orchestration."""

    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose")
    NUM_BETAS = 10
    NUM_JOINTS = 24
    JOINTS = SKEL_JOINTS
    has_head = True

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
        *,
        runtime: Runtime,
    ) -> None:
        if gender not in ("male", "female"):
            raise ValueError(f"Invalid gender: {gender!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        weights = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        self._runtime = runtime
        self._config = SkelConfig(gender=gender)
        self.weights = runtime.convert_model_data(weights)

    @property
    def gender(self) -> Literal["male", "female"]:
        return self._config.gender

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self.weights.faces

    @property
    def skeleton_faces(self) -> Int[Array, "Fs 3"]:
        return self.weights.skel_faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[Array, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def pose_dim(self) -> int:
        return SKEL_CANONICAL_POSE_DIM

    @property
    def body_pose_dim(self) -> int:
        return SKEL_BODY_POSE_DIM

    @property
    def head_pose_dim(self) -> int:
        return SKEL_HEAD_POSE_DIM

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 43"],
        head_pose: Float[Array, "*batch 3"],
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.SkelIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed SKEL vertices."""
        xp = self._runtime.xp
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
            joint_indices=self.weights.skin_joint_indices,
            joint_weights=self.weights.skin_joint_weights,
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
        joint_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 10"] | None = None,
        identity: core.SkelIdentity | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Compute posed SKEL joint transforms."""
        xp = self._runtime.xp
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
            all_axes=self.weights.all_axes,
            rotation_indices=self.weights.rotation_indices,
            apose_R=self.weights.apose_R,
            apose_t=self.weights.apose_t,
            per_joint_rot=self.weights.per_joint_rot,
            child=self.weights.child,
            fixed_orientation_joints=self.weights.fixed_orientation_joints,
            scapula_r_axes=self.weights.scapula_r_axes,
            scapula_l_axes=self.weights.scapula_l_axes,
            spine_axes=self.weights.spine_axes,
            parents=self.weights.parents,
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
            self.weights.v_template,
            self.weights.shapedirs,
            self.weights.j_template,
            self.weights.j_shapedirs,
            self.weights.parent,
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
            all_axes=self.weights.all_axes,
            rotation_indices=self.weights.rotation_indices,
            apose_R=self.weights.apose_R,
            apose_t=self.weights.apose_t,
            per_joint_rot=self.weights.per_joint_rot,
            child=self.weights.child,
            fixed_orientation_joints=self.weights.fixed_orientation_joints,
            scapula_r_axes=self.weights.scapula_r_axes,
            scapula_l_axes=self.weights.scapula_l_axes,
            spine_axes=self.weights.spine_axes,
            parents=self.weights.parents,
            num_joints_smpl=self.weights.num_joints_smpl,
            posedirs=self.weights.posedirs,
            pose=packed_pose,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
            xp=self._runtime.xp,
        )

    def _prepare_skeleton_identity(self, shape: Array) -> core.SkelSkeletonIdentity:
        return core.prepare_skeleton_identity(
            self.weights.j_template,
            self.weights.j_shapedirs,
            self.weights.parent,
            shape,
            xp=self._runtime.xp,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype: Any | None = None) -> dict[str, Array]:
        """Return zero shape and pose controls."""
        runtime = self._runtime
        return {
            "shape": runtime.zeros((*batch_dims, self.NUM_BETAS), like=self.weights.v_template, dtype=dtype),
            "body_pose": runtime.zeros(
                (*batch_dims, self.body_pose_dim),
                like=self.weights.v_template,
                dtype=dtype,
            ),
            "head_pose": runtime.zeros(
                (*batch_dims, self.head_pose_dim),
                like=self.weights.v_template,
                dtype=dtype,
            ),
            "global_rotation": runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype),
            "global_translation": runtime.zeros((*batch_dims, 3), like=self.weights.v_template, dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Array]:
        """Return the SKEL T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Array]:
        """Return the SKEL A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        pose = self._runtime.asarray(SKEL_BODY_PRESETS["a_pose"], like=params["body_pose"])
        pose = self._runtime.xp.broadcast_to(pose, (*batch_dims, *pose.shape))
        params["body_pose"], params["head_pose"] = unpack_pose(self._runtime.xp, pose)
        return params


__all__ = ["SKELModel", "SkelConfig"]
