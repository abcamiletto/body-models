"""PyTorch backend for SOMA model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common

from body_models.base import SkinnedModel, SkinningPayload
from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from .io import (
    MODEL_TYPE_SPECS,
    load_identity_transfer_data,
    load_model_data_for_lod,
    public_joint_metadata,
)
from body_models.bodies.soma.backends import torch as torch_backend
from body_models.bodies.soma.backends import core
from body_models.bodies.soma import identities
from body_models.bodies.soma.identities import torch as identity_sources
from body_models.bodies.soma.constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.bodies.soma.pose import pack_pose, unpack_pose

PathLike = Path | str
__all__ = ["SOMA"]


class SOMA(SkinnedModel, nn.Module):
    """SOMA body model with PyTorch backend."""

    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")
    has_hands = True
    has_head = True
    SHAPE_DIM = 128
    NUM_JOINTS = 77
    VALID_MODEL_TYPES = tuple(MODEL_TYPE_SPECS)
    kernels = ("torch", "warp")
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
        kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        """Initialize the SOMA model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            model_type: SOMA identity/model variant to load.
            lod: Body mesh level of detail: "mid", "low", or "xlo".
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            match_warp: Whether to match Warp backend numerical conventions.
            kernel: Backend kernel used for forward evaluation.
        """
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        super().__init__()

        self.model_type = normalized_model_type
        self.lod = lod.lower()
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.match_warp = match_warp
        self._kernel = _get_kernel(kernel)
        resolved_path, weights = load_model_data_for_lod(model_path, self.lod, simplify=simplify)

        self.weights = torch_backend.prepare_data(common.torchify(weights))
        self.parents, self._joint_names = public_joint_metadata(weights)

        spec = MODEL_TYPE_SPECS[self.model_type]
        self.identity_dim = spec.identity_dim
        self.num_scale_params = spec.num_scale_params
        self._default_identity_value = spec.default_identity_value
        self._identity_source = None
        if spec.asset_dir is not None:
            transfer_data = load_identity_transfer_data(resolved_path, self.model_type)
            self._identity_source = identity_sources.create_identity_source(self.model_type, transfer_data)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def mean_active(self) -> Float[Tensor, "Va 3"]:
        return self.weights.mean_active

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return int(self.weights.mean_active.shape[0])

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        if self.weights.public is not None:
            return self.weights.public.skin_weights_active[:, 1:]
        return self._skinning_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.mean_active * 0.01

    @property
    def _skinning_weights(self) -> Float[Tensor, "V J"]:
        return core.skinning_weights(self.weights)

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
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        *,
        shape: Float[Tensor, "*batch I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        identity: core.SomaIdentity | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            shape: Identity coefficients.
            scale_params: Per-part scale parameters.
            identity: Optional output from :meth:`prepare_identity`.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = tuple(body_pose.shape[: -(self.num_rot_dims + 1)])
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = torch.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity)
        return self._kernel.forward_vertices(
            data=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=pose["skinning_transforms"],
            pose_offsets=pose["pose_offsets"],
            xp=torch,
        )

    def capture_forward_vertices(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        *,
        identity: core.SomaIdentity,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> _CapturedSOMAForward:
        """Capture fixed-shape inference for repeated poses of one identity.

        The returned callable reuses its output tensor. Clone the result before the
        next call if it must be retained.
        """
        if self._kernel is not torch_backend:
            raise ValueError("CUDA graph capture requires kernel='torch'.")
        inputs = (body_pose, head_pose, hand_pose, global_rotation, global_translation)
        return _CapturedSOMAForward(self, inputs, identity, vertex_indices)

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        *,
        shape: Float[Tensor, "*batch I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        identity: core.SomaIdentity | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices: Any | None = None,
    ) -> Float[Tensor, "B 77 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            shape: Identity coefficients.
            scale_params: Per-part scale parameters.
            identity: Optional output from :meth:`prepare_identity`.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = tuple(body_pose.shape[: -(self.num_rot_dims + 1)])
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            if scale_params is not None:
                scale_params = torch.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
            identity = self.prepare_identity(shape, scale_params=scale_params, skip_vertices=True)
        pose = self.prepare_pose(body_pose, head_pose, hand_pose, identity=identity, skip_vertices=True)
        return self._kernel.forward_skeleton(
            data=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            skeleton_transforms=pose["skeleton_transforms"],
            xp=torch,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        device = self.weights.mean_active.device
        pose_ref = torch.zeros((*batch_dims, self.num_joints, 3), device=device, dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=torch,
        )
        global_rotation, body_pose, head_pose, hand_pose = unpack_pose(torch, pose)
        if hands != "default":
            preset = SOMA_HAND_PRESETS[hands]
            axis_angle = torch.asarray(preset, device=device, dtype=dtype).reshape(-1, 3)
            axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        params = {
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }
        params["shape"] = torch.full(
            (*batch_dims, self.identity_dim),
            self._default_identity_value,
            device=device,
            dtype=dtype,
        )
        if self.num_scale_params is not None:
            params["scale_params"] = torch.zeros(
                (*batch_dims, self.num_scale_params),
                device=device,
                dtype=dtype,
            )
        return params

    def prepare_identity(
        self,
        shape: Float[Tensor, "*batch I"],
        *,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        skip_vertices: bool = False,
        repose: bool = True,
        bind_pose: core.BindPoseMode = "fit",
    ) -> core.SomaIdentity:
        """Precompute identity-dependent SOMA state for repeated forward passes."""
        if self.num_scale_params is None:
            scale_params = None
        elif scale_params is None:
            scale_params = torch.zeros(
                (*shape.shape[:-1], self.num_scale_params),
                device=shape.device,
                dtype=shape.dtype,
            )
        return self._prepare_identity_from_inputs(
            shape,
            scale_params,
            skip_vertices=skip_vertices,
            repose=repose,
            bind_pose=bind_pose,
        )

    def prepare_pose(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        *,
        identity: core.SomaIdentity,
        skip_vertices: bool = False,
    ) -> core.SomaPreparedPose:
        """Precompute local pose-dependent state for repeated forward passes."""
        batch_shape = tuple(body_pose.shape[: -(self.num_rot_dims + 1)])
        root_rotation = SO3.identity_as(body_pose, batch_dims=batch_shape, rotation_type=self.rotation_type, xp=torch)
        pose = pack_pose(torch, root_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.prepare_pose(
            self.weights,
            pose,
            rotation_type=self.rotation_type,
            local_joint_translations=identity["local_joint_translations"],
            inverse_bind_transforms=None if skip_vertices else identity["inverse_bind_transforms"],
            skip_vertices=skip_vertices,
            xp=torch,
        )

    def _prepare_identity_from_inputs(
        self,
        shape: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
        *,
        skip_vertices: bool = False,
        repose: bool = True,
        bind_pose: core.BindPoseMode = "fit",
    ) -> core.SomaIdentity:
        rest_shape_full, rest_shape_active = identities.rest_shapes(
            data=self.weights,
            identity_source=self._identity_source,
            identity=shape,
            scale_params=scale_params,
            xp=torch,
        )
        return self._kernel.prepare_identity_from_rest_shape(
            data=self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=torch,
            skip_vertices=skip_vertices,
            repose=repose,
            bind_pose=bind_pose,
        )

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = torch.as_tensor(
            SOMA_BODY_PRESETS["a_pose"],
            device=params["body_pose"].device,
            dtype=params["body_pose"].dtype,
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return params


class _CapturedSOMAForward:
    """Fixed-shape CUDA graph for :meth:`SOMA.forward_vertices`."""

    def __init__(
        self,
        model: SOMA,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None],
        identity: core.SomaIdentity,
        vertex_indices: Any | None,
    ) -> None:
        self._model = model
        self._identity = identity
        self._inputs = tuple(None if value is None else value.detach().clone() for value in inputs)
        self._vertex_indices = (
            None if vertex_indices is None else torch.as_tensor(vertex_indices, device=inputs[0].device)
        )

        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.no_grad(), torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._run()
        current_stream.wait_stream(warmup_stream)

        self._graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self._graph):
            self._output = self._run()

    def __call__(
        self,
        body_pose: Tensor,
        head_pose: Tensor,
        hand_pose: Tensor,
        global_rotation: Tensor | None = None,
        *,
        global_translation: Tensor | None = None,
    ) -> Tensor:
        values = (body_pose, head_pose, hand_pose, global_rotation, global_translation)
        with torch.no_grad():
            for target, value in zip(self._inputs, values, strict=True):
                if target is None and value is None:
                    continue
                if target is None or value is None:
                    raise ValueError("Captured optional inputs must remain present or absent.")
                target.copy_(value)
            self._graph.replay()
        return self._output

    def _run(self) -> Tensor:
        body_pose, head_pose, hand_pose, global_rotation, global_translation = self._inputs
        assert body_pose is not None and head_pose is not None and hand_pose is not None
        return self._model.forward_vertices(
            body_pose,
            head_pose,
            hand_pose,
            global_rotation,
            identity=self._identity,
            global_translation=global_translation,
            vertex_indices=self._vertex_indices,
        )


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.bodies.soma.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use SOMA kernel='warp'.") from exc

    return warp_backend
