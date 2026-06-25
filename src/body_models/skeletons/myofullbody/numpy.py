"""NumPy backend for the MyoFullBody musculoskeletal model."""

from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int

from body_models.base import RigidBodyModel
from trimesh import Trimesh
from body_models.skeletons.myofullbody.backends import core
from body_models.skeletons.myofullbody.backends import numpy as backend
from body_models.skeletons.myofullbody.io import load_model_data
from body_models.skeletons.myofullbody.constants import (
    MYOFULLBODY_BODY_PRESETS,
    MYOFULLBODY_JOINTS,
)

__all__ = ["MyoFullBody"]


class MyoFullBody(RigidBodyModel):
    """MyoSuite-derived full-body MJCF model with rigid STL link meshes."""

    JOINTS = MYOFULLBODY_JOINTS

    def __init__(self, model_path: Path | str | None = None) -> None:
        """Initialize the MyoFullBody model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
        """
        self.weights = load_model_data(model_path)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
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
    def qpos_joint_names(self) -> list[str]:
        return self.weights.qpos_joint_names

    @property
    def qpos_joint_types(self) -> list[str]:
        return self.weights.qpos_joint_types

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
    def link_geom_positions(self) -> Float[np.ndarray, "L 3"]:
        return self.weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[np.ndarray, "L 3 3"]:
        return self.weights.link_geom_rotations

    @property
    def site_names(self) -> list[str]:
        return self.weights.site_names

    @property
    def site_positions(self) -> Float[np.ndarray, "S 3"]:
        return self.weights.site_positions

    @property
    def site_body_indices(self) -> list[int]:
        return self.weights.site_body_indices

    @property
    def tendons(self) -> list[dict]:
        return self.weights.tendons

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    @property
    def num_qpos(self) -> int:
        return self.weights.qpos_joint_axes.shape[0]

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: Any | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local body joint rotations.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return backend.forward_skeleton(
            weights=self.weights,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
        )

    def forward_meshes(
        self,
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        link_indices: Any | None = None,
    ) -> list[Trimesh]:
        """Compute posed link meshes.

        Args:
            body_pose: Local body joint rotations.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            link_indices: Optional subset of links to return.

        Returns:
            One posed mesh payload per link.
        """
        return backend.forward_meshes(
            weights=self.weights,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            link_indices=link_indices,
        )

    def forward_links(
        self,
        body_pose: Float[np.ndarray, "B Q"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
    ) -> Float[np.ndarray, "B L 4 4"]:
        return backend.forward_links(
            weights=self.weights,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
        )

    def world_sites(self, skeleton: Float[np.ndarray, "B J 4 4"]) -> Float[np.ndarray, "B S 3"]:
        return backend.world_sites(self.weights, skeleton)

    def link_mesh(self, link_name: str) -> Trimesh:
        return core.link_mesh(
            vertices=self.weights.vertices,
            faces=self.weights.faces,
            link_joint_indices=self.weights.link_joint_indices,
            link_vertex_starts=self.weights.link_vertex_starts,
            link_vertex_counts=self.weights.link_vertex_counts,
            link_face_starts=self.weights.link_face_starts,
            link_face_counts=self.weights.link_face_counts,
            joint_names=self.weights.joint_names,
            link_names=self.weights.link_names,
            link_name=link_name,
        )

    def joint_meshes(self, joint_name: str) -> list[Trimesh]:
        return core.joint_meshes(
            vertices=self.weights.vertices,
            faces=self.weights.faces,
            link_joint_indices=self.weights.link_joint_indices,
            link_vertex_starts=self.weights.link_vertex_starts,
            link_vertex_counts=self.weights.link_vertex_counts,
            link_face_starts=self.weights.link_face_starts,
            link_face_counts=self.weights.link_face_counts,
            joint_names=self.weights.joint_names,
            link_names=self.weights.link_names,
            joint_name=joint_name,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        return {
            "body_pose": np.zeros((*batch_dims, self.num_qpos), dtype=dtype),
            "global_rotation": np.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = np.asarray(MYOFULLBODY_BODY_PRESETS["t_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = np.broadcast_to(body_pose, (*batch_dims, *body_pose.shape)).copy()
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = np.asarray(MYOFULLBODY_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = np.broadcast_to(body_pose, (*batch_dims, *body_pose.shape)).copy()
        return params
