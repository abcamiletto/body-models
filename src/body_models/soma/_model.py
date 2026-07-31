"""SOMA model implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import ParameterSpec, SkinnedModel, SkinningIdentity, SkinningPayload, SkinningPose
from body_models._common import skinning
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import ArrayRuntime, RuntimeLike
from body_models.anny import ANNY
from body_models.mhr import MHR
from body_models.smpl import SMPL
from body_models.smplx import SMPLX
from body_models.soma import _core as core
from body_models.soma import _identities as identities
from body_models.soma._constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.soma._io import (
    MODEL_TYPE_SPECS,
    load_identity_transfer_data,
    load_model_data_for_lod,
    public_joint_metadata,
)
from body_models.soma._pose import pack_pose

Array = Any
PathLike = Path | str
SomaIdentity = core.SomaIdentity
SomaPreparedPose = core.SomaPreparedPose
_IdentityModel = ANNY | MHR | SMPL | SMPLX
_IDENTITY_MODEL_CLASSES: dict[str, Callable[..., _IdentityModel]] = {
    "anny": ANNY,
    "mhr": MHR,
    "smpl": SMPL,
    "smplx": SMPLX,
}


@dataclass(frozen=True)
class SomaConfig:
    """Static SOMA behavior kept outside array state."""

    model_type: str
    lod: str
    rotation_type: RotationType
    identity_dim: int
    num_scale_params: int | None
    default_identity_value: float
    parents: tuple[int, ...]
    joint_names: tuple[str, ...]


class SOMA(SkinnedModel):
    """Native SOMA-X model with identity, pose, and corrective controls."""

    _state_fields = ("_weights", "_identity_model", "_identity_transfer")
    has_hands = True
    NUM_JOINTS = 77
    NUM_BODY_JOINTS = 23
    NUM_HAND_JOINTS = 48
    NUM_HEAD_JOINTS = 5
    JOINTS = SOMA_JOINTS

    def __init__(
        self,
        *,
        model_path: PathLike | None = None,
        model_type: str = "soma",
        lod: str = "mid",
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in MODEL_TYPE_SPECS:
            supported = ", ".join(MODEL_TYPE_SPECS)
            raise ValueError(f"Invalid model_type: {model_type!r}. Expected one of {supported}.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        normalized_lod = lod.lower()
        resolved_path, weights = load_model_data_for_lod(model_path, normalized_lod, simplify=simplify)
        spec = MODEL_TYPE_SPECS[normalized_model_type]
        parents, joint_names = public_joint_metadata(weights)
        runtime = self._set_runtime(runtime)
        self._config = SomaConfig(
            model_type=normalized_model_type,
            lod=normalized_lod,
            rotation_type=rotation_type,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
            parents=tuple(parents),
            joint_names=tuple(joint_names),
        )
        self._weights = runtime.materialize(weights)
        self._identity_model = None
        self._identity_transfer = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, normalized_model_type)
            self._identity_model = _create_identity_model(normalized_model_type, runtime)
            transfer = identities.prepare_transfer(
                normalized_model_type,
                transfer_data,
                self._identity_model,
                runtime,
            )
            self._identity_transfer = runtime.materialize(transfer)

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
    def identity_dim(self) -> int:
        return self._config.identity_dim

    @property
    def num_scale_params(self) -> int | None:
        return self._config.num_scale_params

    @property
    def _num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        parameters = {
            "shape": ParameterSpec(
                (self.identity_dim,),
                "identity",
                default=self._config.default_identity_value,
            ),
        }
        if self.num_scale_params is not None:
            parameters["scale_params"] = ParameterSpec((self.num_scale_params,), "identity")
        parameters.update(
            {
                "body_pose": ParameterSpec.rotation(rotation, count=self.NUM_BODY_JOINTS),
                "head_pose": ParameterSpec.rotation(rotation, count=self.NUM_HEAD_JOINTS),
                "hand_pose": ParameterSpec.rotation(rotation, count=self.NUM_HAND_JOINTS),
                "global_rotation": ParameterSpec.rotation(rotation, role="transform"),
                "global_translation": ParameterSpec((3,), "transform"),
            }
        )
        return parameters

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self._config.joint_names)

    @property
    def parents(self) -> list[int]:
        return list(self._config.parents)

    @property
    def num_vertices(self) -> int:
        return self._weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        if self._weights.public is not None:
            return self._weights.public.skin_weights_active[:, 1:]
        return self._skinning_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.mean_active * 0.01

    @property
    def _skinning_weights(self) -> Float[Array, "V J"]:
        return self._weights.skin_weights_active[:, 1:]

    def prepare_skinning(
        self,
        *,
        identity: SkinningIdentity,
        pose: SkinningPose,
    ) -> SkinningPayload:
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
        *,
        shape: Float[Array, "*batch I"] | None = None,
        scale_params: Float[Array, "*batch K"] | None = None,
        identity: SomaIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Int[Array, "S"] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute posed mesh vertices in meters."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, scale_params=scale_params)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = xp.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params)

        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
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
            self.rotation_type,
            xp=xp,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        *,
        shape: Float[Array, "*batch I"] | None = None,
        scale_params: Float[Array, "*batch K"] | None = None,
        identity: SomaIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch 77 4 4"]:
        """Compute posed public-joint transforms in meters."""
        xp = self._runtime.xp
        self._validate_identity_arguments(identity, shape=shape, scale_params=scale_params)
        if identity is None:
            if shape is None:
                raise ValueError("shape is required when identity is not provided")
            batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
            shape = xp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = xp.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            skeleton_identity = self._prepare_skeleton_identity(shape, scale_params=scale_params)
        else:
            skeleton_identity = identity

        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        pose = pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            self._weights,
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
    ) -> SomaIdentity:
        """Precompute identity-dependent state for repeated forward passes."""
        rest_shape_full, rest_shape_active = self._rest_shapes(shape, scale_params)
        return core.prepare_identity_from_rest_shape(
            data=self._weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            runtime=self._runtime,
            repose=repose,
            bind_pose=bind_pose,
        )

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        *,
        identity: SomaIdentity,
    ) -> SomaPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        xp = self._runtime.xp
        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        pose = pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            self._weights,
            pose,
            rotation_type=self.rotation_type,
            local_joint_translations=identity["local_joint_translations"],
            inverse_bind_transforms=identity["inverse_bind_transforms"],
            xp=xp,
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch I"],
        *,
        scale_params: Float[Array, "*batch K"] | None,
    ) -> core.SomaSkeletonIdentity:
        rest_shape_full, rest_shape_active = self._rest_shapes(shape, scale_params)
        return core.prepare_skeleton_identity_from_rest_shape(
            self._weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            runtime=self._runtime,
        )

    def _rest_shapes(
        self,
        shape: Float[Array, "*batch I"],
        scale_params: Float[Array, "*batch K"] | None,
    ) -> tuple[Float[Array, "*batch Vf 3"], Float[Array, "*batch Va 3"]]:
        if self.num_scale_params is None:
            scale_params = None
        elif scale_params is None:
            scale_params = self._runtime.zeros(
                (*shape.shape[:-1], self.num_scale_params),
                like=shape,
            )
        return identities.rest_shapes(
            data=self._weights,
            model_type=self.model_type,
            identity_model=self._identity_model,
            identity_transfer=self._identity_transfer,
            identity=shape,
            scale_params=scale_params,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero pose controls and the model's default identity."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            runtime = self.runtime
            axis_angle = runtime.asarray(SOMA_HAND_PRESETS[hands], like=params["hand_pose"]).reshape(-1, 3)
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
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SOMA T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SOMA A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        xp = self._runtime.xp
        axis_angle = self._runtime.asarray(SOMA_BODY_PRESETS["a_pose"], like=params["body_pose"])
        axis_angle = xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=xp)
        return params


def _create_identity_model(model_type: str, runtime: ArrayRuntime) -> _IdentityModel:
    spec = MODEL_TYPE_SPECS[model_type]
    model_class = _IDENTITY_MODEL_CLASSES[model_type]
    return model_class(
        simplify=1.0,
        runtime=runtime,
        **spec.identity_model_kwargs,
    )


__all__ = ["SOMA", "SomaConfig"]
