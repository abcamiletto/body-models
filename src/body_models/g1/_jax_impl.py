"""JAX implementation for the Unitree G1 rigid model."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import load_model_data

__all__ = ["G1"]


class G1(BodyModel, nnx.Module):
    """Unitree G1 as rigid STL links attached to the Kimodo 34-joint skeleton."""

    NUM_JOINTS = 34

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "rotmat",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        data = load_model_data(model_path)
        self._joint_names = data["joint_names"]
        self.parents = data["parents"]
        self.link_names = data["link_names"]
        self.qpos_joint_names = data["qpos_joint_names"]
        for key in [
            "local_offsets",
            "rest_local_rotations",
            "rest_joints",
            "vertices",
            "skin_weights",
            "link_geom_positions",
            "link_geom_rotations",
            "qpos_joint_axes",
            "qpos_joint_limits",
        ]:
            setattr(self, key if key not in {"vertices", "skin_weights"} else f"_{key}", nnx.Variable(jnp.asarray(data[key])))
        for key in [
            "faces",
            "link_joint_indices",
            "link_vertex_starts",
            "link_vertex_counts",
            "link_face_starts",
            "link_face_counts",
            "qpos_joint_indices",
        ]:
            setattr(self, f"_{key}", nnx.Variable(jnp.asarray(np.asarray(data[key], dtype=np.int64))))

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self._vertices[...].shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self._skin_weights[...]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        params = self.get_rest_pose(batch_size=1, dtype=self._vertices[...].dtype)
        return self.forward_vertices(**params)[0]

    def forward_skeleton(
        self,
        pose: Float[jax.Array, "B 34 N"] | Float[jax.Array, "B 34 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B 34 4 4"]:
        return core.forward_skeleton(
            local_offsets=self.local_offsets[...],
            rest_local_rotations=self.rest_local_rotations[...],
            parents=self.parents,
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=jnp,
        )

    def forward_vertices(
        self,
        pose: Float[jax.Array, "B 34 N"] | Float[jax.Array, "B 34 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        vertex_indices=None,
        return_per_link: bool = False,
    ):
        return core.forward_vertices(
            vertices=self._vertices[...],
            faces=self._faces[...],
            local_offsets=self.local_offsets[...],
            rest_local_rotations=self.rest_local_rotations[...],
            parents=self.parents,
            link_joint_indices=self._link_joint_indices[...],
            link_vertex_starts=self._link_vertex_starts[...],
            link_vertex_counts=self._link_vertex_counts[...],
            link_face_starts=self._link_face_starts[...],
            link_face_counts=self._link_face_counts[...],
            link_geom_positions=self.link_geom_positions[...],
            link_geom_rotations=self.link_geom_rotations[...],
            link_names=self.link_names,
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            return_per_link=return_per_link,
            rotation_type=self.rotation_type,
            xp=jnp,
        )

    def project_pose_to_qpos(
        self,
        pose: Float[jax.Array, "B 34 N"] | Float[jax.Array, "B 34 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        clamp_to_limits: bool = True,
    ) -> Float[jax.Array, "B Q"]:
        return core.project_pose_to_qpos(
            qpos_joint_indices=self._qpos_joint_indices[...],
            qpos_joint_axes=self.qpos_joint_axes[...],
            qpos_joint_limits=self.qpos_joint_limits[...],
            pose=pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            clamp_to_limits=clamp_to_limits,
            rotation_type=self.rotation_type,
            xp=jnp,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        return {
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
