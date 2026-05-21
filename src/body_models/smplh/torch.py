"""PyTorch backend for SMPL-H model."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplh.backends import torch as torch_backend
from body_models.smplh.backends.core import SmplhIdentity, SmplhPreparedPose
from body_models.smplh.io import get_model_path, load_model_data
from body_models.smplh.constants import SMPLH_BODY_PRESETS, SMPLH_HAND_PRESETS, SMPLH_JOINTS

__all__ = ["SMPLH"]


class SMPLH(BodyModel, nn.Module):
    """SMPL-H body model with PyTorch backend."""

    has_hands = True

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_JOINTS = 52
    kernels = ("torch", "warp")
    JOINTS = SMPLH_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["torch", "warp"] = "torch",
    ):
        """Initialize the SMPLH model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            flat_hand_mean: Whether to use a flat hand as the pose mean.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        super().__init__()

        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

        resolved_path = get_model_path(model_path, gender)
        weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)
        self.weights = common.torchify(weights)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 52"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[Tensor, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[Tensor, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[Tensor, "V 52"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[Tensor, "*batch 21 N"] | Float[Tensor, "*batch 21 3 3"],
        hand_pose: Float[Tensor, "*batch 30 N"] | Float[Tensor, "*batch 30 3 3"],
        pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
        global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 10"] | None = None,
        identity: SmplhIdentity | None = None,
    ) -> Float[Tensor, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)
        pose = self.prepare_pose(body_pose, hand_pose, pelvis_rotation, identity=identity)
        assert "rest_vertices" in identity
        assert "pose_offsets" in pose
        return self._kernel.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            rest_joints=identity["rest_joints"],
            rest_vertices=identity["rest_vertices"],
            joint_transforms=pose["joint_transforms"],
            pose_offsets=pose["pose_offsets"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "*batch 21 N"] | Float[Tensor, "*batch 21 3 3"],
        hand_pose: Float[Tensor, "*batch 30 N"] | Float[Tensor, "*batch 30 3 3"],
        pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
        global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
        global_translation: Float[Tensor, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 10"] | None = None,
        identity: SmplhIdentity | None = None,
    ) -> Float[Tensor, "*batch 52 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        pose = self.prepare_pose(body_pose, hand_pose, pelvis_rotation, identity=identity, skip_vertices=True)
        return self._kernel.forward_skeleton(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            joint_transforms=pose["joint_transforms"],
        )

    def prepare_identity(
        self,
        shape: Float[Tensor, "*batch 10"],
        skip_vertices: bool = False,
    ) -> SmplhIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return self._kernel.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[Tensor, "*batch 21 N"] | Float[Tensor, "*batch 21 3 3"],
        hand_pose: Float[Tensor, "*batch 30 N"] | Float[Tensor, "*batch 30 3 3"],
        pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
        *,
        identity: SmplhIdentity,
        skip_vertices: bool = False,
    ) -> SmplhPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return self._kernel.prepare_pose(
            self.weights,
            body_pose,
            hand_pose,
            pelvis_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=torch.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        device = self.rest_vertices.device
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        body_pose_ref = torch.zeros((*batch_dims, self.NUM_BODY_JOINTS, 3), device=device, dtype=dtype)
        hand_pose_ref = torch.zeros((*batch_dims, self.NUM_HAND_JOINTS, 3), device=device, dtype=dtype)
        pelvis_ref = torch.zeros((*batch_dims, 3), device=device, dtype=dtype)
        params = {
            "shape": torch.zeros((*batch_dims, 10), device=device, dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(*batch_dims, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_dims, device, dtype, hands)
        return params

    def _hand_preset(self, batch_dims: tuple[int, ...], device, dtype: torch.dtype, hands: str):
        preset = SMPLH_HAND_PRESETS[hands]
        axis_angle = torch.as_tensor(preset, device=device, dtype=dtype).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = torch.as_tensor(
            SMPLH_BODY_PRESETS["a_pose"],
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
        from body_models.smplh.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use SMPLH kernel='warp'.") from exc

    return warp_backend
