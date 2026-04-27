"""PyTorch implementation for the MyoFullBody musculoskeletal model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from ..base import BodyModel
from . import core
from .io import load_model_data

__all__ = ["MyoFullBody"]


class MyoFullBody(BodyModel, nn.Module):
    """MyoSuite-derived full-body MJCF model with rigid STL link meshes."""

    local_offsets: Tensor
    rest_local_rotations: Tensor
    qpos_axes: Tensor
    qpos_anchors: Tensor
    qpos_limits: Tensor
    hinge_mask: Tensor
    slide_mask: Tensor
    link_geom_positions: Tensor
    link_geom_rotations: Tensor
    _vertices: Tensor
    _faces: Tensor

    def __init__(self, model_path: Path | str | None = None) -> None:
        super().__init__()
        data = load_model_data(model_path)
        self._joint_names = data["joint_names"]
        self.parents = data["parents"]
        self.qpos_joint_names = data["qpos_joint_names"]
        self.qpos_joint_types = data["qpos_joint_types"]
        self.body_qpos_starts = data["body_qpos_starts"]
        self.body_qpos_counts = data["body_qpos_counts"]
        self.link_names = data["link_names"]
        self.link_joint_indices = data["link_joint_indices"]
        self.link_vertex_starts = data["link_vertex_starts"]
        self.link_vertex_counts = data["link_vertex_counts"]
        self.link_face_starts = data["link_face_starts"]
        self.link_face_counts = data["link_face_counts"]

        buffer_map = {
            "local_offsets": data["local_offsets"],
            "rest_local_rotations": data["rest_local_rotations"],
            "qpos_axes": data["qpos_joint_axes"],
            "qpos_anchors": data["qpos_joint_anchors"],
            "qpos_limits": data["qpos_joint_limits"],
            "hinge_mask": data["hinge_mask"],
            "slide_mask": data["slide_mask"],
            "link_geom_positions": data["link_geom_positions"],
            "link_geom_rotations": data["link_geom_rotations"],
            "site_positions": data["site_positions"],
        }
        for name, value in buffer_map.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer("_vertices", torch.as_tensor(data["vertices"], dtype=torch.float32))
        self.register_buffer("_faces", torch.as_tensor(data["faces"], dtype=torch.int64))
        self.site_names = data["site_names"]
        self.site_body_indices = data["site_body_indices"]
        self.tendons = data["tendons"]

    # ------------------------------------------------------------------
    # BodyModel interface
    # ------------------------------------------------------------------

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self._vertices.shape[0]

    @property
    def num_qpos(self) -> int:
        return self.qpos_axes.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        params = self.get_rest_pose(batch_size=1, dtype=self._vertices.dtype)
        return self.forward_vertices(**params)[0]

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B J 4 4"]:
        return core.forward_skeleton(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            parents=self.parents,
            body_qpos_starts=self.body_qpos_starts,
            body_qpos_counts=self.body_qpos_counts,
            qpos_axes=self.qpos_axes,
            qpos_anchors=self.qpos_anchors,
            hinge_mask=self.hinge_mask,
            slide_mask=self.slide_mask,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            xp=torch,
        )

    def forward_vertices(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        return core.forward_vertices(
            vertices=self._vertices,
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            parents=self.parents,
            body_qpos_starts=self.body_qpos_starts,
            body_qpos_counts=self.body_qpos_counts,
            qpos_axes=self.qpos_axes,
            qpos_anchors=self.qpos_anchors,
            hinge_mask=self.hinge_mask,
            slide_mask=self.slide_mask,
            link_joint_indices=self.link_joint_indices,
            link_vertex_starts=self.link_vertex_starts,
            link_vertex_counts=self.link_vertex_counts,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            xp=torch,
        )

    def forward_links(
        self,
        body_pose: Float[Tensor, "B Q"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        *,
        global_rotation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B L 4 4"]:
        return core.forward_links(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            parents=self.parents,
            body_qpos_starts=self.body_qpos_starts,
            body_qpos_counts=self.body_qpos_counts,
            qpos_axes=self.qpos_axes,
            qpos_anchors=self.qpos_anchors,
            hinge_mask=self.hinge_mask,
            slide_mask=self.slide_mask,
            link_joint_indices=self.link_joint_indices,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            xp=torch,
        )

    def link_mesh(self, link_name: str) -> dict[str, Tensor | str | int]:
        return core.link_mesh(
            vertices=self._vertices,
            faces=self._faces,
            link_joint_indices=self.link_joint_indices,
            link_vertex_starts=self.link_vertex_starts,
            link_vertex_counts=self.link_vertex_counts,
            link_face_starts=self.link_face_starts,
            link_face_counts=self.link_face_counts,
            joint_names=self._joint_names,
            link_names=self.link_names,
            link_name=link_name,
        )

    def joint_meshes(self, joint_name: str) -> list[dict[str, Tensor | str | int]]:
        return core.joint_meshes(
            vertices=self._vertices,
            faces=self._faces,
            link_joint_indices=self.link_joint_indices,
            link_vertex_starts=self.link_vertex_starts,
            link_vertex_counts=self.link_vertex_counts,
            link_face_starts=self.link_face_starts,
            link_face_counts=self.link_face_counts,
            joint_names=self._joint_names,
            link_names=self.link_names,
            joint_name=joint_name,
        )

    def world_sites(self, skeleton: Float[Tensor, "B J 4 4"]) -> Float[Tensor, "B S 3"]:
        """World-space site positions for a given ``forward_skeleton`` output."""
        return core.world_sites(
            skeleton=skeleton,
            site_positions=self.site_positions,
            site_body_indices=self.site_body_indices,
            xp=torch,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self._vertices.device
        return {
            "body_pose": torch.zeros((batch_size, self.num_qpos), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
