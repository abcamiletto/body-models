"""ANNY model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import ParameterSpec, PointRegressor, SkinnedModel, SkinningPose
from body_models._common import skinning
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import ArrayRuntime
from body_models.anny import _core as core
from body_models.anny import _pose as pose_utils
from body_models.anny._constants import ANNY_BODY_PRESETS, ANNY_HAND_PRESETS, ANNY_JOINTS
from body_models.anny._io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy

Array = Any
HandPreset = Literal["default", "flat", "rest"]
AnnyIdentity = core.AnnyIdentity


@dataclass(frozen=True)
class AnnyConfig:
    """Static ANNY behavior preserved outside array state."""

    all_phenotypes: bool
    extrapolate_phenotypes: bool
    rotation_type: RotationType


class ANNY(SkinnedModel):
    """Phenotype-driven skinned body model."""

    has_hands = True
    NUM_BODY_JOINTS = 64
    NUM_HAND_JOINTS = 38
    NUM_HEAD_JOINTS = 60
    NUM_SHAPE_COEFFS = 6
    _COMMON_JOINTS = ANNY_JOINTS
    _POSE_LAYOUT = pose_utils.POSE_LAYOUT

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        rig: Literal["default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"] = "default",
        topology: Literal["default", "makehuman"] = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: ArrayRuntime,
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
        self._attach_runtime(runtime)
        self._config = AnnyConfig(
            all_phenotypes=all_phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            rotation_type=rotation_type,
        )
        self._weights = runtime._materialize(weights)

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
    def _num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity", default=0.5),
            "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_JOINTS),
            "head_pose": ParameterSpec.rotation(rotation, count=self.NUM_HEAD_JOINTS),
            "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_JOINTS),
            "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    @property
    def phenotype_labels(self) -> list[str]:
        if self.all_phenotypes:
            return list(PHENOTYPE_LABELS)
        return [label for label in PHENOTYPE_LABELS if label not in EXCLUDED_PHENOTYPES]

    @property
    def faces(self) -> Int[Array, "F _"]:
        return self._weights.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self._weights.template_vertices.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.template_vertices

    @property
    def parents(self) -> list[int]:
        return list(self._weights.kinematic_tree.parents)

    @property
    def _skinning_triangles(self) -> Int[Array, "F 3"]:
        return self._weights.triangles

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"],
        head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"],
        hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"],
        *,
        shape: Float[Array, "*batch 6"] | None = None,
        identity: AnnyIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed ANNY vertices."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        vertices = self._runtime._skin_vertices(
            identity["rest_vertices"],
            pose["skinning_transforms"],
            skinning=self._weights.compact_skinning,
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
        *,
        shape: Float[Array, "*batch 6"] | None = None,
        identity: AnnyIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed ANNY joint transforms."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape)
        else:
            skeleton_identity = identity

        batch_shape = tuple(body_pose.shape[: -(self._num_rot_dims + 1)])
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        packed_pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            self._runtime,
            self._weights.kinematic_tree,
            packed_pose,
            self.rotation_type,
            rest_skeleton_transforms=skeleton_identity["rest_skeleton_transforms"],
        )
        return skinning.transform_skeleton(
            skeleton,
            global_rotation,
            global_translation,
            self.rotation_type,
            joint_indices,
            xp=xp,
        )

    def forward_points(
        self,
        body_pose: Float[Array, "*batch 64 N"] | Float[Array, "*batch 64 3 3"],
        head_pose: Float[Array, "*batch 60 N"] | Float[Array, "*batch 60 3 3"],
        hand_pose: Float[Array, "*batch 38 N"] | Float[Array, "*batch 38 3 3"],
        *,
        point_regressor: PointRegressor,
        shape: Float[Array, "*batch 6"] | None = None,
        identity: AnnyIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch K 3"]:
        """Compute positions defined by a prepared vertex mapping."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

    def prepare_identity(
        self,
        shape: Float[Array, "*batch 6"],
    ) -> AnnyIdentity:
        """Precompute phenotype-dependent state for repeated forward passes."""
        return core.prepare_identity(
            xp=self._runtime.xp,
            template_vertices=self._weights.template_vertices,
            blendshapes=self._weights.blendshapes,
            template_bone_heads=self._weights.template_bone_heads,
            template_bone_tails=self._weights.template_bone_tails,
            bone_heads_blendshapes=self._weights.bone_heads_blendshapes,
            bone_tails_blendshapes=self._weights.bone_tails_blendshapes,
            bone_rolls_rotmat=self._weights.bone_rolls_rotmat,
            phenotype_mask=self._weights.phenotype_mask,
            anchors=self._weights.anchors,
            y_axis=self._weights.y_axis,
            degenerate_rotation=self._weights.degenerate_rotation,
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
        identity: AnnyIdentity,
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        xp = self._runtime.xp
        batch_shape = tuple(body_pose.shape[: -(self._num_rot_dims + 1)])
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        packed_pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            self._runtime,
            self._weights.kinematic_tree,
            packed_pose,
            self.rotation_type,
            rest_skeleton_transforms=identity["rest_skeleton_transforms"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch 6"],
    ) -> core.AnnySkeletonIdentity:
        return core.prepare_skeleton_identity(
            xp=self._runtime.xp,
            template_bone_heads=self._weights.template_bone_heads,
            template_bone_tails=self._weights.template_bone_tails,
            bone_heads_blendshapes=self._weights.bone_heads_blendshapes,
            bone_tails_blendshapes=self._weights.bone_tails_blendshapes,
            bone_rolls_rotmat=self._weights.bone_rolls_rotmat,
            phenotype_mask=self._weights.phenotype_mask,
            anchors=self._weights.anchors,
            y_axis=self._weights.y_axis,
            degenerate_rotation=self._weights.degenerate_rotation,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            shape=shape,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return centered phenotype controls and identity rotations."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")

        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            runtime = self.runtime
            axis_angle = runtime.asarray(ANNY_HAND_PRESETS[hands], like=params["hand_pose"]).reshape(-1, 3)
            axis_angle = runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            params["hand_pose"] = SO3.convert(
                axis_angle,
                src="axis_angle",
                dst=self.rotation_type,
                xp=runtime.xp,
            )
        return params

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the ANNY T-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
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
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: HandPreset = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the ANNY rest A-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)


__all__ = ["ANNY", "AnnyConfig"]
