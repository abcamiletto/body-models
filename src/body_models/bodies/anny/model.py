"""Single-source ANNY model program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import SkinnedModel, SkinningPayload
from body_models.bodies.anny import core, pose as pose_utils
from body_models.bodies.anny.constants import ANNY_BODY_PRESETS, ANNY_HAND_PRESETS, ANNY_JOINTS
from body_models.bodies.anny.io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy
from body_models.common import skinning
from body_models.rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models.runtime import Runtime

Array = Any
HandPreset = Literal["default", "flat", "rest"]


@dataclass(frozen=True)
class AnnyConfig:
    """Static ANNY behavior preserved outside array state."""

    all_phenotypes: bool
    extrapolate_phenotypes: bool
    rotation_type: RotationType


class ANNYModel(SkinnedModel):
    """Backend-independent ANNY interface and orchestration."""

    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")
    has_hands = True
    has_head = True
    JOINTS = ANNY_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        runtime: Runtime,
    ) -> None:
        if rig not in ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"):
            raise ValueError(f"Invalid rig: {rig!r}")
        if topology not in ("default", "makehuman"):
            raise ValueError(f"Invalid topology: {topology!r}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        weights = load_model_data_numpy(model_path, rig=rig, topology=topology, simplify=simplify)
        self._runtime = runtime
        self._config = AnnyConfig(
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            rotation_type=rotation_type,
        )
        self.weights = runtime.convert_model_data(weights)

    @property
    def all_phenotypes(self) -> bool:
        return self._config.all_phenotypes

    @property
    def extrapolate_phenotypes(self) -> bool:
        return self._config.extrapolate_phenotypes

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def phenotype_labels(self) -> list[str]:
        if self.all_phenotypes:
            return list(PHENOTYPE_LABELS)
        return [label for label in PHENOTYPE_LABELS if label not in EXCLUDED_PHENOTYPES]

    @property
    def faces(self) -> Int[Array, "F _"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.bone_labels)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.weights.template_vertices.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.template_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def prepare_skinning(self, *, identity: Mapping[str, Any], pose: Mapping[str, Any]) -> SkinningPayload:
        payload = super().prepare_skinning(identity=identity, pose=pose)
        payload["faces"] = _triangulate_faces(self.faces, self._runtime.xp)
        return payload

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"],
        head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"],
        hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 6"] | None = None,
        identity: core.AnnyIdentity | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed ANNY vertices."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"],
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
        body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"],
        head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"],
        hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Array, "*batch 6"] | None = None,
        identity: core.AnnyIdentity | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed ANNY joint transforms."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape)
        else:
            skeleton_identity = identity

        batch_shape = tuple(body_pose.shape[: -(self.num_rot_dims + 1)])
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        packed_pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            self.weights.kinematic_fronts,
            packed_pose,
            self.rotation_type,
            rest_skeleton_transforms=skeleton_identity["rest_skeleton_transforms"],
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
        shape: Float[Array, "*batch 6"],
    ) -> core.AnnyIdentity:
        """Precompute phenotype-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            template_vertices=self.weights.template_vertices,
            blendshapes=self.weights.blendshapes,
            template_bone_heads=self.weights.template_bone_heads,
            template_bone_tails=self.weights.template_bone_tails,
            bone_heads_blendshapes=self.weights.bone_heads_blendshapes,
            bone_tails_blendshapes=self.weights.bone_tails_blendshapes,
            bone_rolls_rotmat=self.weights.bone_rolls_rotmat,
            phenotype_mask=self.weights.phenotype_mask,
            anchors=self.weights.anchors,
            y_axis=self.weights.y_axis,
            degenerate_rotation=self.weights.degenerate_rotation,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            shape=shape,
        )

    def phenotype_to_shape(
        self,
        gender: Float[Array, "*batch"],
        age: Float[Array, "*batch"],
        muscle: Float[Array, "*batch"],
        weight: Float[Array, "*batch"],
        height: Float[Array, "*batch"],
        proportions: Float[Array, "*batch"],
    ) -> Float[Array, "*batch 6"]:
        """Pack named phenotype controls into the ANNY shape vector."""
        return self._runtime.xp.stack([gender, age, muscle, weight, height, proportions], axis=-1)

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"],
        head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"],
        hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"],
        *,
        identity: core.AnnyIdentity,
    ) -> core.AnnyPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        xp = self._runtime.xp
        batch_shape = tuple(body_pose.shape[: -(self.num_rot_dims + 1)])
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        packed_pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            self.weights.kinematic_fronts,
            packed_pose,
            self.rotation_type,
            rest_skeleton_transforms=identity["rest_skeleton_transforms"],
            xp=xp,
        )

    def _prepare_skeleton_identity(self, shape: Array) -> core.AnnySkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            template_bone_heads=self.weights.template_bone_heads,
            template_bone_tails=self.weights.template_bone_tails,
            bone_heads_blendshapes=self.weights.bone_heads_blendshapes,
            bone_tails_blendshapes=self.weights.bone_tails_blendshapes,
            bone_rolls_rotmat=self.weights.bone_rolls_rotmat,
            phenotype_mask=self.weights.phenotype_mask,
            anchors=self.weights.anchors,
            y_axis=self.weights.y_axis,
            degenerate_rotation=self.weights.degenerate_rotation,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            shape=shape,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Array]:
        """Return centered phenotype controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        runtime = self._runtime
        pose_ref = runtime.zeros(batch_dims, like=self.weights.template_vertices, dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=runtime.xp,
        )
        global_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(runtime.xp, pose)
        if hands != "default":
            axis_angle = runtime.asarray(ANNY_HAND_PRESETS[hands], like=hand_pose).reshape(-1, 3)
            axis_angle = runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(
                axis_angle,
                src="axis_angle",
                dst=self.rotation_type,
                xp=runtime.xp,
            )
        shape = runtime.zeros((*batch_dims, 6), like=self.weights.template_vertices, dtype=dtype) + 0.5
        return {
            "shape": shape,
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": runtime.zeros(
                (*batch_dims, 3),
                like=self.weights.template_vertices,
                dtype=dtype,
            ),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return the ANNY T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = self._runtime.asarray(ANNY_BODY_PRESETS["t_pose"], like=params["body_pose"])
        axis_angle = self._runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst=self.rotation_type,
            xp=self._runtime.xp,
        )
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: HandPreset = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return the ANNY rest A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


def _triangulate_faces(faces: Int[Array, "F _"], xp: Any) -> Int[Array, "Ftri 3"]:
    if faces.shape[-1] == 3:
        return faces
    return xp.concat([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)


__all__ = ["ANNYModel", "AnnyConfig"]
