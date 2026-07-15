"""Shared program surface for rigid articulated models."""

from __future__ import annotations

from typing import Any

from jaxtyping import Float, Int
from trimesh import Trimesh

from body_models.base import RigidBodyModel
from body_models.common import rigid
from body_models.runtime import Runtime

Array = Any


class RigidModel(RigidBodyModel):
    """Common rigid-model state, metadata, and mesh projection."""

    weights: Any
    _runtime: Runtime

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self.weights.faces

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
    def actuated_joint_limits(self) -> Float[Array, "Q 2"]:
        return self.weights.actuated_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def link_joint_indices(self) -> list[int]:
        return self.weights.link_joint_indices

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def _link_transforms(self, skeleton: Array) -> Array:
        return rigid.forward_link_transforms(
            skeleton,
            self.weights.link_joint_indices,
            self.weights.link_geom_positions,
            self.weights.link_geom_rotations,
            xp=self._runtime.xp,
        )

    def _meshes_from_links(self, links: Array) -> list[Trimesh]:
        return rigid.forward_meshes_from_links(
            links,
            self.weights.vertices,
            self.weights.faces,
            self.weights.link_vertex_starts,
            self.weights.link_vertex_counts,
            self.weights.link_face_starts,
            self.weights.link_face_counts,
            xp=self._runtime.xp,
        )

    def _zero_pose(self, pose_key: str, batch_dims: tuple[int, ...], dtype: Any | None) -> dict[str, Array]:
        runtime = self._runtime
        reference = self.weights.vertices
        return {
            pose_key: runtime.zeros((*batch_dims, self.num_actuated), like=reference, dtype=dtype),
            "global_rotation": runtime.zeros((*batch_dims, 3), like=reference, dtype=dtype),
            "global_translation": runtime.zeros((*batch_dims, 3), like=reference, dtype=dtype),
        }


__all__ = ["RigidModel"]
