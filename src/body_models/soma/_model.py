"""SOMA model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models._base import (
    CorrectiveBasis,
    ParameterSpec,
    PointRegressor,
    SkinnedModel,
    SkinningPose,
    SparseCorrectiveBasis,
)
from body_models._common import skinning
from body_models._registry import create_model
from body_models._rotations import VALID_ROTATION_TYPES, RotationType, rotation_ndim
from body_models._runtime import ArrayRuntime, TorchRuntime
from body_models.soma import _core as core
from body_models.soma import _identities as identities
from body_models.soma import _pose as pose_utils
from body_models.soma._constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.soma._io import (
    MODEL_TYPE_SPECS,
    SOMA_LODS,
    load_identity_transfer_data,
    load_model_data_for_lod,
)

Array = Any
PathLike = Path | str
SomaIdentity = core.SomaIdentity


@dataclass(frozen=True)
class SomaConfig:
    """Static SOMA behavior kept outside array state."""

    model_type: Literal["soma", "anny", "mhr", "smpl", "smplx"]
    lod: Literal["mid", "low", "xlo"]
    rotation_type: RotationType
    num_shape_coeffs: int
    num_scale_coeffs: int | None
    default_identity_value: float


class SOMA(SkinnedModel):
    """Native SOMA-X model with identity, pose, and corrective controls."""

    _state_fields = ("_weights", "_identity_model", "_identity_transfer")
    has_hands = True
    NUM_JOINTS = 77
    NUM_BODY_JOINTS = 23
    NUM_HAND_JOINTS = 48
    NUM_HEAD_JOINTS = 5
    _COMMON_JOINTS = SOMA_JOINTS
    _POSE_LAYOUT = pose_utils.POSE_LAYOUT

    def __init__(
        self,
        *,
        model_path: PathLike | None = None,
        model_type: Literal["soma", "anny", "mhr", "smpl", "smplx"] = "soma",
        lod: Literal["mid", "low", "xlo"] = "mid",
        rotation_type: RotationType = "axis_angle",
        simplify: float = 1.0,
        runtime: ArrayRuntime,
    ) -> None:
        if model_type not in MODEL_TYPE_SPECS:
            supported = ", ".join(MODEL_TYPE_SPECS)
            raise ValueError(f"Invalid model_type: {model_type!r}. Expected one of {supported}.")
        if lod not in SOMA_LODS:
            supported = ", ".join(SOMA_LODS)
            raise ValueError(f"Invalid lod: {lod!r}. Expected one of {supported}.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type!r}")

        resolved_path, weights = load_model_data_for_lod(model_path, lod, simplify=simplify)
        spec = MODEL_TYPE_SPECS[model_type]
        self._attach_runtime(runtime)
        self._config = SomaConfig(
            model_type=model_type,
            lod=lod,
            rotation_type=rotation_type,
            num_shape_coeffs=spec.num_shape_coeffs,
            num_scale_coeffs=spec.num_scale_coeffs,
            default_identity_value=spec.default_identity_value,
        )
        self._weights = runtime._materialize(weights)
        self._identity_model = None
        self._identity_transfer = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, model_type)
            self._identity_model = _create_identity_model(model_type, runtime)
            transfer = identities.prepare_transfer(
                model_type,
                transfer_data,
                self._identity_model,
                runtime,
            )
            self._identity_transfer = runtime._materialize(transfer)

    @property
    def model_type(self) -> Literal["soma", "anny", "mhr", "smpl", "smplx"]:
        return self._config.model_type

    @property
    def lod(self) -> Literal["mid", "low", "xlo"]:
        return self._config.lod

    @property
    def rotation_type(self) -> RotationType:
        return self._config.rotation_type

    @property
    def num_shape_coeffs(self) -> int:
        return self._config.num_shape_coeffs

    @property
    def num_scale_coeffs(self) -> int | None:
        return self._config.num_scale_coeffs

    @property
    def _num_rot_dims(self) -> int:
        return rotation_ndim(self.rotation_type)

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        rotation = self.rotation_type
        parameters = {
            "shape": ParameterSpec(
                (self.num_shape_coeffs,),
                "identity",
                default=self._config.default_identity_value,
            ),
        }
        if self.num_scale_coeffs is not None:
            parameters["scale_params"] = ParameterSpec((self.num_scale_coeffs,), "identity")
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
    def joint_names(self) -> list[str]:
        return list(self._weights.control_rig.joint_names_full[1:])

    @property
    def parents(self) -> list[int]:
        parents = self._weights.control_rig.kinematics.kinematic_tree.parents
        return [parent - 1 for parent in parents[1:]]

    @property
    def num_vertices(self) -> int:
        return self._weights.mean_active.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self._weights.control_rig.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self._weights.mean_active * 0.01

    @property
    def _skinning_weights(self) -> Float[Array, "V J"]:
        return self._weights.skin_weights_active[:, 1:]

    @property
    def _corrective_basis(self) -> CorrectiveBasis:
        return SparseCorrectiveBasis(self._weights.correctives.basis)

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
        vertex_indices: Sequence[int] | None = None,
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
        joint_indices: Sequence[int] | None = None,
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
        pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        skeleton = core.prepare_skeleton(
            self._runtime,
            self._weights,
            pose,
            self.rotation_type,
            local_joint_translations=skeleton_identity["local_joint_translations"],
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
        body_pose: Float[Array, "*batch 23 N"] | Float[Array, "*batch 23 3 3"],
        head_pose: Float[Array, "*batch 5 N"] | Float[Array, "*batch 5 3 3"],
        hand_pose: Float[Array, "*batch 48 N"] | Float[Array, "*batch 48 3 3"],
        *,
        point_regressor: PointRegressor,
        shape: Float[Array, "*batch I"] | None = None,
        scale_params: Float[Array, "*batch K"] | None = None,
        identity: SomaIdentity | None = None,
        global_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch P 3"]:
        """Compute positions defined by a prepared vertex mapping."""
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
        return self._deform_points(point_regressor, identity, pose, global_rotation, global_translation)

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
            self._runtime,
            data=self._weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
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
    ) -> SkinningPose:
        """Precompute pose-dependent state for repeated forward passes."""
        xp = self._runtime.xp
        batch_shape = body_pose.shape[: -(self._num_rot_dims + 1)]
        root_rotation = SO3.identity_as(
            body_pose,
            batch_dims=batch_shape,
            rotation_type=self.rotation_type,
            xp=xp,
        )
        pose = pose_utils.pack_pose(xp, root_rotation, body_pose, head_pose, hand_pose)
        return core.prepare_pose(
            self._runtime,
            self._weights,
            pose,
            rotation_type=self.rotation_type,
            local_joint_translations=identity["local_joint_translations"],
            inverse_bind_transforms=identity["inverse_bind_transforms"],
        )

    def _prepare_skeleton_identity(
        self,
        shape: Float[Array, "*batch I"],
        *,
        scale_params: Float[Array, "*batch K"] | None,
    ) -> core.SomaSkeletonIdentity:
        rest_shape_full, rest_shape_active = self._rest_shapes(shape, scale_params)
        return core.prepare_skeleton_identity_from_rest_shape(
            self._runtime,
            self._weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
        )

    def _rest_shapes(
        self,
        shape: Float[Array, "*batch I"],
        scale_params: Float[Array, "*batch K"] | None,
    ) -> tuple[Float[Array, "*batch Vf 3"], Float[Array, "*batch Va 3"]]:
        if self.num_scale_coeffs is None:
            scale_params = None
        elif scale_params is None:
            scale_params = self._runtime.zeros(
                (*shape.shape[:-1], self.num_scale_coeffs),
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
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SOMA T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SOMA A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        xp = self._runtime.xp
        axis_angle = self._runtime.asarray(SOMA_BODY_PRESETS["a_pose"], like=params["body_pose"])
        axis_angle = xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=xp)
        return params


def _create_identity_model(model_type: str, runtime: ArrayRuntime) -> Any:
    spec = MODEL_TYPE_SPECS[model_type]
    kwargs = dict(spec.identity_model_kwargs) | {"simplify": 1.0}
    if isinstance(runtime, TorchRuntime):
        kwargs["kernel_backend"] = runtime.kernel_backend
    return create_model(model_type, runtime=runtime.name, **kwargs)


__all__ = ["SOMA", "SomaConfig"]
