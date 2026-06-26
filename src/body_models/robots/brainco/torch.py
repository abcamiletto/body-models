"""PyTorch backend for the BrainCo Revo 2 robotic hand model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from body_models import common
from body_models.base import RigidBodyModel
from trimesh import Trimesh
from body_models.robots.brainco.backends import core
from body_models.robots.brainco.backends import torch as backend
from body_models.robots.brainco.io import Side, load_model_data
from body_models.robots.brainco.constants import BRAINCO_HAND_PRESETS, LEFT_BRAINCO_JOINTS, RIGHT_BRAINCO_JOINTS

__all__ = ["BrainCoHand"]


class BrainCoHand(RigidBodyModel, nn.Module):
    """BrainCo Revo 2 as rigid STL links attached to its MuJoCo hand skeleton."""

    has_hands = True
    mujoco_to_model = core.MUJOCO_TO_KIMODO

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        side: Side = "right",
    ) -> None:
        """Initialize the BrainCoHand model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            side: Hand side to load.
        """
        super().__init__()
        self.weights = common.torchify(load_model_data(model_path, side=side))

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def side(self) -> Side:
        return self.weights.side

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def common_joints(self):
        return LEFT_BRAINCO_JOINTS if self.side == "left" else RIGHT_BRAINCO_JOINTS

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def actuated_joint_names(self) -> list[str]:
        return self.weights.actuated_joint_names

    @property
    def actuated_joint_limits(self) -> Float[Tensor, "Q 2"]:
        return self.weights.actuated_joint_limits

    @property
    def actuated_joint_types(self) -> list[str]:
        return ["hinge"] * self.num_actuated

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def link_joint_indices(self) -> list[int]:
        return self.weights.link_joint_indices

    @property
    def link_vertex_starts(self) -> list[int]:
        return self.weights.link_vertex_starts

    @property
    def link_vertex_counts(self) -> list[int]:
        return self.weights.link_vertex_counts

    @property
    def link_face_starts(self) -> list[int]:
        return self.weights.link_face_starts

    @property
    def link_face_counts(self) -> list[int]:
        return self.weights.link_face_counts

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        hand_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            hand_pose: Local hinge coordinates.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return backend.forward_skeleton(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
        )

    def forward_meshes(
        self,
        hand_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> list[Trimesh]:
        """Compute posed model meshes.

        Args:
            hand_pose: Local hinge coordinates.
            global_translation: Global model translation.
            global_rotation: Global model rotation.

        Returns:
            One posed model mesh per batch element.
        """
        return backend.forward_meshes(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def forward_links(
        self,
        hand_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        device = self.weights.vertices.device
        hand_pose = torch.zeros((self.num_actuated,), device=device, dtype=dtype)
        if hands != "default":
            hand_pose = torch.as_tensor(BRAINCO_HAND_PRESETS[self.side][hands], device=device, dtype=dtype)
        hand_pose = torch.broadcast_to(hand_pose, (*batch_dims, self.num_actuated))
        return {
            "hand_pose": hand_pose,
            "global_rotation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }
