"""JAX backend for the SMPL humanoid robot."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from body_models import common
from body_models.base import RigidBodyModel
from body_models.robots.smpl_humanoid.backends import core
from body_models.robots.smpl_humanoid.backends import jax as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import load_model_data

__all__ = ["SmplHumanoid"]


class SmplHumanoid(RigidBodyModel):
    """Rigid humanoid loaded from MJCF using the canonical 24-joint SMPL hierarchy."""

    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self, model_path: Path | str | None = None, *, rotation_type: core.RotationType = "axis_angle"
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type for SmplHumanoid: {rotation_type}")
        self.rotation_type = rotation_type
        self.global_rotation_type = rotation_type
        self.weights = common.jaxify(load_model_data(model_path))

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
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
    def actuated_joint_limits(self) -> Float[jax.Array, "Q 2"]:
        return self.weights.actuated_joint_limits

    @property
    def actuated_joint_types(self) -> list[str]:
        return self.weights.actuated_joint_types

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
    def link_geom_positions(self) -> Float[jax.Array, "L 3"]:
        return self.weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[jax.Array, "L 3 3"]:
        return self.weights.link_geom_rotations

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "B Q"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_meshes(
        self,
        body_pose: Float[jax.Array, "B Q"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    ) -> list[Trimesh]:
        return backend.forward_meshes(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        body_pose: Float[jax.Array, "B Q"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=jnp.float32) -> dict[str, jax.Array]:
        global_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        return {
            "body_pose": jnp.zeros((*batch_dims, self.num_actuated), dtype=dtype),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = jnp.asarray(SMPL_BODY_PRESETS[name], dtype=params["body_pose"].dtype)
        ordered = jnp.concat([axis_angle[smpl_index] for _, smpl_index in BODY_JOINTS])
        params["body_pose"] = jnp.broadcast_to(ordered, (*batch_dims, ordered.shape[0]))
        return params
