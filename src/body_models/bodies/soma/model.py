"""Single-source SOMA model program."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import SkinnedModel, SkinningPayload
from body_models.bodies.soma import core, identities
from body_models.bodies.soma.constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.bodies.soma.io import (
    MODEL_TYPE_SPECS,
    load_identity_transfer_data,
    load_model_data_for_lod,
    public_joint_metadata,
)
from body_models.bodies.soma.lowerings import SomaLowerings
from body_models.bodies.soma.pose import pack_pose, unpack_pose
from body_models.common import skinning
from body_models.rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models.runtime import Runtime

Array = Any
PathLike = Path | str


@dataclass(frozen=True)
class SomaConfig:
    """Static SOMA behavior kept outside array state."""

    model_type: str
    lod: str
    rotation_type: RotationType
    match_warp: bool
    identity_dim: int
    num_scale_params: int | None
    default_identity_value: float


class SOMAModel(SkinnedModel):
    """Backend-independent SOMA interface and orchestration."""

    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")
    has_hands = True
    has_head = True
    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)
    JOINTS = SOMA_JOINTS

    def __init__(
        self,
        model_path: PathLike | None = None,
        *,
        model_type: str = "soma",
        lod: str = "mid",
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        runtime: Runtime,
        lowerings: SomaLowerings,
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            supported = ", ".join(self.VALID_MODEL_TYPES)
            raise ValueError(f"Invalid model_type: {model_type!r}. Expected one of {supported}.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        normalized_lod = lod.lower()
        resolved_path, weights = load_model_data_for_lod(model_path, normalized_lod, simplify=simplify)
        spec = MODEL_TYPE_SPECS[normalized_model_type]
        self._runtime = runtime
        self._config = SomaConfig(
            model_type=normalized_model_type,
            lod=normalized_lod,
            rotation_type=rotation_type,
            match_warp=match_warp,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
        )
        self.parents, self._joint_names = public_joint_metadata(weights)
        self.weights = runtime.convert_model_data(weights)
        self._corrective_network = lowerings.corrective_network(runtime, self.weights)
        self._identity_source = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, normalized_model_type)
            self._identity_source = lowerings.identity_source(
                normalized_model_type,
                transfer_data,
            )

    @property
    def model_type(self) -> str:
        return self._config.model_type

    @property
    def lod(self) -> str:
        return self._config.lod

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def match_warp(self) -> bool:
        return self._config.match_warp

    @property
    def identity_dim(self) -> int:
        return self._config.identity_dim

    @property
    def num_scale_params(self) -> int | None:
        return self._config.num_scale_params

    @property
    def num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self.weights.faces

    @property
    def mean_active(self) -> Float[Array, "Va 3"]:
        return self.weights.mean_active

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        if self.weights.public is not None:
            return self.weights.public.skin_weights_active[:, 1:]
        return self._skinning_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.weights.mean_active * 0.01

    @property
    def _skinning_weights(self) -> Float[Array, "V J"]:
        return self.weights.skin_weights_active[:, 1:]

    def prepare_skinning(self, *, identity: Mapping[str, Any], pose: Mapping[str, Any]) -> SkinningPayload:
        return {
            "rest_vertices": identity["rest_vertices"],
            "skinning_transforms": pose["skinning_transforms"],
            "pose_offsets": pose["pose_offsets"],
            "skin_weights": self._skinning_weights,
            "faces": self.faces,
        }

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        *,
        shape: Float[Array, "*batch I"] | None = None,
        scale_params: Float[Array, "*batch K"] | None = None,
        identity: core.SomaIdentity | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices in meters."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = xp.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        vertices = self._runtime.compact_linear_blend_skinning(
            identity["rest_vertices"] + pose["pose_offsets"],
            pose["skinning_transforms"],
            joint_indices=self.weights.skin_joint_indices_active - 1,
            joint_weights=self.weights.skin_joint_weights_active,
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
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        *,
        shape: Float[Array, "*batch I"] | None = None,
        scale_params: Float[Array, "*batch K"] | None = None,
        identity: core.SomaIdentity | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch 77 4 4"]:
        """Compute posed public-joint transforms in meters."""
        xp = self._runtime.xp
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = xp.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape, scale_params=scale_params)
        else:
            skeleton_identity = identity

        batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        pose = pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            self.weights,
            pose,
            self.rotation_type,
            local_joint_translations=skeleton_identity["local_joint_translations"],
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
        shape: Float[Array, "*batch I"],
        *,
        scale_params: Float[Array, "*batch K"] | None = None,
        repose: bool = True,
        bind_pose: core.BindPoseMode = "fit",
    ) -> core.SomaIdentity:
        """Precompute identity-dependent state for repeated forward passes."""
        rest_shape_full, rest_shape_active = self._rest_shapes(shape, scale_params)
        return core.prepare_identity_from_rest_shape(
            data=self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=self._runtime.xp,
            repose=repose,
            bind_pose=bind_pose,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        *,
        identity: core.SomaIdentity,
    ) -> core.SomaPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        xp = self._runtime.xp
        batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        pose = pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            self.weights,
            pose,
            rotation_type=self.rotation_type,
            local_joint_translations=identity["local_joint_translations"],
            inverse_bind_transforms=identity["inverse_bind_transforms"],
            xp=xp,
            corrective_network=self._corrective_network,
        )

    def _prepare_skeleton_identity(
        self,
        shape: Array,
        *,
        scale_params: Array | None,
    ) -> core.SomaSkeletonIdentity:
        rest_shape_full, rest_shape_active = self._rest_shapes(shape, scale_params)
        return core.prepare_skeleton_identity_from_rest_shape(
            self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=self._runtime.xp,
        )

    def _rest_shapes(self, shape: Array, scale_params: Array | None) -> tuple[Array, Array]:
        if self.num_scale_params is None:
            scale_params = None
        elif scale_params is None:
            scale_params = self._runtime.zeros(
                (*shape.shape[:-1], self.num_scale_params),
                like=shape,
            )
        return identities.rest_shapes(
            data=self.weights,
            identity_source=self._identity_source,
            identity=shape,
            scale_params=scale_params,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Array]:
        """Return zero pose controls and the model's default identity."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        runtime = self._runtime
        xp = runtime.xp
        pose_reference = runtime.zeros((*batch_dims, self.num_joints, 3), like=self.weights.mean_active, dtype=dtype)
        pose = SO3.identity_as(
            pose_reference,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=xp,
        )
        global_rotation, body_pose, head_pose, hand_pose = unpack_pose(xp, pose)
        if hands != "default":
            axis_angle = runtime.asarray(SOMA_HAND_PRESETS[hands], like=pose_reference).reshape(-1, 3)
            axis_angle = xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=xp)

        params = {
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": runtime.zeros((*batch_dims, 3), like=pose_reference),
            "shape": runtime.zeros((*batch_dims, self.identity_dim), like=pose_reference)
            + self._config.default_identity_value,
        }
        if self.num_scale_params is not None:
            params["scale_params"] = runtime.zeros(
                (*batch_dims, self.num_scale_params),
                like=pose_reference,
            )
        return params

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return the SOMA T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return the SOMA A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        xp = self._runtime.xp
        axis_angle = self._runtime.asarray(SOMA_BODY_PRESETS["a_pose"], like=params["body_pose"])
        axis_angle = xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=xp)
        return params


__all__ = ["SOMAModel", "SomaConfig"]
