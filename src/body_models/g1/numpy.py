"""NumPy backend for the Unitree G1 rigid model."""

__all__ = ["G1"]

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from . import core
from .io import load_model_data



class G1(BodyModel):
    """Unitree G1 as rigid STL links attached to the Kimodo 34-joint skeleton."""

    NUM_JOINTS = 34

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "rotmat",
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        data = load_model_data(model_path)
        self._joint_names = data["joint_names"]
        self.parents = data["parents"]
        self.local_offsets = data["local_offsets"]
        self.rest_local_rotations = data["rest_local_rotations"]
        self._vertices = data["vertices"]
        self._faces = data["faces"]
        self.link_joint_indices = data["link_joint_indices"]
        self.link_vertex_starts = data["link_vertex_starts"]
        self.link_vertex_counts = data["link_vertex_counts"]
        self.link_face_starts = data["link_face_starts"]
        self.link_face_counts = data["link_face_counts"]
        self.link_geom_positions = data["link_geom_positions"]
        self.link_geom_rotations = data["link_geom_rotations"]
        self.link_names = data["link_names"]
        self.qpos_joint_indices = data["qpos_joint_indices"]
        self.qpos_joint_axes = data["qpos_joint_axes"]
        self.qpos_joint_limits = data["qpos_joint_limits"]
        self.qpos_joint_names = data["qpos_joint_names"]

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self._vertices.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        params = self.get_rest_pose(batch_size=1)
        return self.forward_vertices(
            body_pose=params["body_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )[0]

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "B 29 N"] | Float[np.ndarray, "B 29 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 34 4 4"]:
        return core.forward_skeleton(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            body_joint_indices=self.qpos_joint_indices,
            body_joint_axes=self.qpos_joint_axes,
            parents=self.parents,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def forward_vertices(
        self,
        body_pose: Float[np.ndarray, "B 29 N"] | Float[np.ndarray, "B 29 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return core.forward_vertices(
            vertices=self._vertices,
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            body_joint_indices=self.qpos_joint_indices,
            body_joint_axes=self.qpos_joint_axes,
            parents=self.parents,
            link_joint_indices=self.link_joint_indices,
            link_vertex_starts=self.link_vertex_starts,
            link_vertex_counts=self.link_vertex_counts,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def forward_links(
        self,
        body_pose: Float[np.ndarray, "B 29 N"] | Float[np.ndarray, "B 29 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    ) -> Float[np.ndarray, "B L 4 4"]:
        return core.forward_links(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            body_joint_indices=self.qpos_joint_indices,
            body_joint_axes=self.qpos_joint_axes,
            parents=self.parents,
            link_joint_indices=self.link_joint_indices,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def link_mesh(self, link_name: str) -> dict[str, np.ndarray | str | int]:
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

    def joint_meshes(self, joint_name: str) -> list[dict[str, np.ndarray | str | int]]:
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

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((batch_size, len(self.qpos_joint_indices), 3), dtype=dtype)
        global_ref = np.zeros((batch_size, 3), dtype=dtype)
        body_pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, len(self.qpos_joint_indices)),
            rotation_type=self.rotation_type,
            xp=np,
        )
        global_rotation = SO3.identity_as(
            global_ref,
            batch_dims=(batch_size,),
            rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
            xp=np,
        )
        return {
            "body_pose": body_pose,
            "global_rotation": global_rotation,
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
