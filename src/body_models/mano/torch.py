"""PyTorch backend for MANO model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.mano.backends import torch as torch_backend
from body_models.mano.io import get_model_path, load_model_data
from body_models.mano.constants import LEFT_MANO_JOINTS, MANO_HAND_PRESETS, RIGHT_MANO_JOINTS
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["MANO"]


class MANO(BodyModel, nn.Module):
    """MANO hand model with PyTorch backend."""

    has_hands = True

    NUM_HAND_JOINTS = 15
    NUM_JOINTS = 16
    kernels = ("torch", "warp")

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["torch", "warp"] = "torch",
    ):
        """Initialize the MANO model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            side: Hand side to load.
            flat_hand_mean: Whether to use a flat hand as the pose mean.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side}. Must be 'right' or 'left'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        super().__init__()

        self.side = side if side is not None else "right"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

        resolved_path = get_model_path(model_path, side)
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
    def common_joints(self):
        return LEFT_MANO_JOINTS if self.side == "left" else RIGHT_MANO_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 16"]:
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
    def lbs_weights(self) -> Float[Tensor, "V 16"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        hand_pose: Float[Tensor, "B 15 N"] | Float[Tensor, "B 15 3 3"],
        wrist_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            hand_pose: Local hand joint rotations.
            wrist_rotation: Root wrist rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        hand_pose: Float[Tensor, "B 15 N"] | Float[Tensor, "B 15 3 3"],
        wrist_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B 16 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            hand_pose: Local hand joint rotations.
            wrist_rotation: Root wrist rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        device = self.rest_vertices.device
        hand_pose_ref = torch.zeros((*batch_dims, self.NUM_HAND_JOINTS, 3), device=device, dtype=dtype)
        wrist_ref = torch.zeros((*batch_dims, 3), device=device, dtype=dtype)
        hand_pose = SO3.identity_as(
            hand_pose_ref,
            batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
            rotation_type=self.rotation_type,
            xp=torch,
        )
        if hands != "default":
            hand_pose = self._hand_preset(batch_dims, device, dtype, hands)
        return {
            "shape": torch.zeros((*batch_dims, 10), device=device, dtype=dtype),
            "hand_pose": hand_pose,
            "wrist_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def _hand_preset(self, batch_dims: tuple[int, ...], device, dtype: torch.dtype, hands: str):
        preset = MANO_HAND_PRESETS[self.side][hands]
        axis_angle = torch.as_tensor(preset, device=device, dtype=dtype).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.mano.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use MANO kernel='warp'.") from exc

    return warp_backend
