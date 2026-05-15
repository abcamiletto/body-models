"""PyTorch backend for SOMA model."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .io import (
    MODEL_TYPE_SPECS,
    compute_sparse_skin_weights,
    get_model_path,
    load_identity_transfer_data,
    load_model_data,
    simplify_mesh,
)
from body_models.soma.backends import torch as torch_backend
from body_models.soma.backends import core
from body_models.soma import identities
from body_models.soma.identities import torch as identity_sources
from body_models.soma.constants import SOMA_BODY_PRESETS, SOMA_HAND_PRESETS, SOMA_JOINTS
from body_models.soma.pose import pack_pose, unpack_pose

PathLike = Path | str
__all__ = ["SOMA"]


class SOMA(BodyModel, nn.Module):
    """SOMA body model with PyTorch backend."""

    has_hands = True

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
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        match_warp: bool = True,
        kernel: Literal["torch", "warp"] = "torch",
    ) -> None:
        normalized_model_type = model_type.lower()
        if normalized_model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: {model_type}. Supported SOMA model types are {', '.join(self.VALID_MODEL_TYPES)}."
            )
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0 (1.0 = original mesh)")
        super().__init__()

        self.model_type = normalized_model_type
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.match_warp = match_warp
        self._kernel = _get_kernel(kernel)
        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        mean_full = data.mean_full
        shapedirs_full = data.shapedirs_full
        faces = data.faces
        skin_weights_full = data.skin_weights_full

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            vertex_map = np.asarray(vertex_map, dtype=np.int64)
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            vertex_map = None

        skin_joint_indices_active, skin_joint_weights_active = compute_sparse_skin_weights(skin_weights_active)
        weights = replace(
            data,
            mean_active=np.asarray(mean_active, dtype=np.float32),
            shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
            skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
            skin_joint_indices_active=skin_joint_indices_active,
            skin_joint_weights_active=skin_joint_weights_active,
            faces=np.asarray(faces, dtype=np.int64),
            vertex_map=vertex_map,
        )
        self.weights = common.torchify(weights)
        self.parents = [parent - 1 for parent in data.topology.parents_full[1:]]
        self._joint_names = data.joint_names_full[1:]

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
        return self.weights.skin_weights_active[:, 1:]

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.mean_active * 0.01

    def forward_vertices(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        *,
        identity: Float[Tensor, "B|1 I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
        prepared_identity: core.PreparedSomaIdentity | None = None,
    ) -> Float[Tensor, "B V 3"]:
        pose = pack_pose(torch, global_rotation, body_pose, head_pose, hand_pose)
        identity_state = prepared_identity
        if identity_state is None:
            identity_state = self.prepare_identity(identity=identity, scale_params=scale_params, pose=pose)
        return self._kernel.forward_vertices(
            data=self.weights,
            prepared_identity=identity_state,
            pose=pose,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
            xp=torch,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
        head_pose: Float[Tensor, "B 5 N"] | Float[Tensor, "B 5 3 3"],
        hand_pose: Float[Tensor, "B 48 N"] | Float[Tensor, "B 48 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        *,
        identity: Float[Tensor, "B|1 I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
        prepared_identity: core.PreparedSomaIdentity | None = None,
    ) -> Float[Tensor, "B 77 4 4"]:
        pose = pack_pose(torch, global_rotation, body_pose, head_pose, hand_pose)
        identity_state = prepared_identity
        if identity_state is None:
            identity_state = self.prepare_identity(identity=identity, scale_params=scale_params, pose=pose)
        return self._kernel.forward_skeleton(
            data=self.weights,
            prepared_identity=identity_state,
            pose=pose,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
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
        params["identity"] = torch.full(
            (*batch_dims, self.identity_dim),
            self._default_identity_value,
            device=device,
            dtype=dtype,
        )
        if self.num_scale_params is not None:
            params["scale_params"] = torch.zeros(
                (1, self.num_scale_params),
                device=device,
                dtype=dtype,
            )
        return params

    def prepare_identity(
        self,
        *,
        identity: Float[Tensor, "B|1 I"] | None = None,
        scale_params: Float[Tensor, "B|1 K"] | None = None,
        pose: Float[Tensor, "B ..."],
    ) -> core.PreparedSomaIdentity:
        identity, scale_params = self._identity_inputs(identity=identity, scale_params=scale_params, pose=pose)
        return self._prepare_identity_from_inputs(identity, scale_params)

    def _identity_inputs(
        self,
        *,
        identity: Float[Tensor, "B|1 I"] | None,
        scale_params: Float[Tensor, "B|1 K"] | None,
        pose: Float[Tensor, "B ..."],
    ) -> tuple[Float[Tensor, "B I"], Float[Tensor, "B K"] | None]:
        pose_ndim = self.num_rot_dims + 1
        batch_shape = tuple(pose.shape[:-pose_ndim])
        if identity is None:
            identity = torch.full(
                (*batch_shape, self.identity_dim),
                self._default_identity_value,
                device=pose.device,
                dtype=pose.dtype,
            )
        elif identity.shape[:-1] == (1,) and batch_shape:
            identity = torch.broadcast_to(identity, (*batch_shape, identity.shape[-1]))

        if self.num_scale_params is None:
            if scale_params is not None:
                raise ValueError("scale_params is only supported for SOMA model_type='mhr'.")
            return identity, None

        if scale_params is None:
            scale_params = torch.zeros(
                (*batch_shape, self.num_scale_params),
                device=identity.device,
                dtype=identity.dtype,
            )
        elif scale_params.shape[:-1] == (1,) and batch_shape:
            scale_params = torch.broadcast_to(scale_params, (*batch_shape, scale_params.shape[-1]))
        return identity, scale_params

    def _prepare_identity_from_inputs(
        self,
        identity: Float[Tensor, "B I"],
        scale_params: Float[Tensor, "B K"] | None,
    ) -> core.PreparedSomaIdentity:
        rest_shape_full, rest_shape_active = identities.rest_shapes(
            data=self.weights,
            identity_source=self._identity_source,
            identity=identity,
            scale_params=scale_params,
            xp=torch,
        )
        return self._kernel.prepare_identity_from_rest_shape(
            data=self.weights,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=self.match_warp,
            xp=torch,
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


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.soma.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use SOMA kernel='warp'.") from exc

    return warp_backend
