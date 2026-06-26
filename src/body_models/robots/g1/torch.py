"""PyTorch backend for the Unitree G1 rigid model."""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.base import RigidBodyModel
from trimesh import Trimesh
from body_models.robots.g1.backends import core
from body_models.robots.g1.backends import torch as backend
from body_models.robots.g1.io import load_model_data
from body_models.robots.g1.constants import G1_BODY_PRESETS, G1_JOINTS

__all__ = ["G1"]


class G1(RigidBodyModel, nn.Module):
    """Unitree G1 as rigid STL links attached to the Kimodo 34-joint skeleton."""

    JOINTS = G1_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
    ) -> None:
        """Initialize the G1 model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            convention: Skeleton convention used when loading rigid model data.
        """
        super().__init__()
        self.mujoco_to_model = core.MUJOCO_TO_KIMODO if convention == "soma" else self.mujoco_to_model
        self.convention = convention
        self.weights = common.torchify(load_model_data(model_path, convention=convention))

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

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
    def link_geom_positions(self) -> Float[Tensor, "L 3"]:
        return self.weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[Tensor, "L 3 3"]:
        return self.weights.link_geom_rotations

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local hinge coordinates.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
        )

    def forward_meshes(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> list[Trimesh]:
        """Compute posed model meshes.

        Args:
            body_pose: Local hinge coordinates.
            global_translation: Global model translation.
            global_rotation: Global model rotation.

        Returns:
            One posed model mesh per batch element.
        """
        return backend.forward_meshes(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def forward_links(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.weights.vertices.device
        return {
            "body_pose": torch.zeros((*batch_dims, self.num_actuated), device=device, dtype=dtype),
            "global_rotation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = torch.as_tensor(
            G1_BODY_PRESETS["t_pose"],
            device=params["body_pose"].device,
            dtype=params["body_pose"].dtype,
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        dst_kwargs = {"axes": self.weights.actuated_joint_axes}
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst="hinge",
            dst_kwargs=dst_kwargs,
            xp=torch,
        )[..., 0]
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = torch.as_tensor(
            G1_BODY_PRESETS["a_pose"],
            device=params["body_pose"].device,
            dtype=params["body_pose"].dtype,
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        dst_kwargs = {"axes": self.weights.actuated_joint_axes}
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst="hinge",
            dst_kwargs=dst_kwargs,
            xp=torch,
        )[..., 0]
        return params
