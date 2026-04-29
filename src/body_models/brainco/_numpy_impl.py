"""NumPy implementation for the BrainCo Revo 2 robotic hand model."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from . import core
from .io import Side, load_model_data

__all__ = ["BrainCoHand"]


class BrainCoHand(BodyModel):
    """BrainCo Revo 2 as rigid STL links attached to its MuJoCo hand skeleton."""

    is_rigid_body = True

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        side: Side = "right",
        rotation_type: core.RotationType = "rotmat",
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        data = load_model_data(model_path, side=side)
        self.side = data["side"]
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
        self.coupled_joint_indices = data["coupled_joint_indices"]
        self.coupled_joint_axes = data["coupled_joint_axes"]
        self.coupled_driver_indices = data["coupled_driver_indices"]
        self.coupled_polycoef = data["coupled_polycoef"]

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
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
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.forward_vertices(**self.get_rest_pose(batch_size=1))[0]

    def forward_skeleton(
        self,
        pose: Float[np.ndarray, "B 6 N"] | Float[np.ndarray, "B 6 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 12 4 4"]:
        return core.forward_skeleton(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            joint_axes=self.qpos_joint_axes,
            joint_indices=self.qpos_joint_indices,
            coupled_joint_axes=self.coupled_joint_axes,
            coupled_joint_indices=self.coupled_joint_indices,
            coupled_driver_indices=self.coupled_driver_indices,
            coupled_polycoef=self.coupled_polycoef,
            parents=self.parents,
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            skeleton_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def forward_vertices(
        self,
        pose: Float[np.ndarray, "B 6 N"] | Float[np.ndarray, "B 6 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return core.forward_vertices(
            vertices=self._vertices,
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            joint_axes=self.qpos_joint_axes,
            joint_indices=self.qpos_joint_indices,
            coupled_joint_axes=self.coupled_joint_axes,
            coupled_joint_indices=self.coupled_joint_indices,
            coupled_driver_indices=self.coupled_driver_indices,
            coupled_polycoef=self.coupled_polycoef,
            parents=self.parents,
            link_joint_indices=self.link_joint_indices,
            link_vertex_starts=self.link_vertex_starts,
            link_vertex_counts=self.link_vertex_counts,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def forward_links(self, pose, global_translation=None, *, global_rotation=None):
        return core.forward_links(
            local_offsets=self.local_offsets,
            rest_local_rotations=self.rest_local_rotations,
            joint_axes=self.qpos_joint_axes,
            joint_indices=self.qpos_joint_indices,
            coupled_joint_axes=self.coupled_joint_axes,
            coupled_joint_indices=self.coupled_joint_indices,
            coupled_driver_indices=self.coupled_driver_indices,
            coupled_polycoef=self.coupled_polycoef,
            parents=self.parents,
            link_joint_indices=self.link_joint_indices,
            link_geom_positions=self.link_geom_positions,
            link_geom_rotations=self.link_geom_rotations,
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def link_mesh(self, link_name: str) -> dict[str, np.ndarray | str]:
        return core.link_mesh(
            self._vertices,
            self._faces,
            self.link_vertex_starts,
            self.link_vertex_counts,
            self.link_face_starts,
            self.link_face_counts,
            self.link_names,
            link_name,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((batch_size, len(self.qpos_joint_indices), 3), dtype=dtype)
        global_ref = np.zeros((batch_size, 3), dtype=dtype)
        return {
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, len(self.qpos_joint_indices)),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=(batch_size,),
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
