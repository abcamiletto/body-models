"""JAX backend for the Unitree G1 rigid model."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.base import BodyModel
from body_models.g1.backends import core
from body_models.g1.backends import jax as backend
from body_models.g1.io import load_model_data
from body_models.g1.constants import G1_APOSE, G1_IPOSE, G1_JOINTS, G1_TPOSE

__all__ = ["G1"]


class G1(BodyModel):
    """Unitree G1 as rigid STL links attached to the Kimodo 34-joint skeleton."""

    is_rigid_body = True
    JOINTS = G1_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "rotmat",
        convention: core.Convention = "soma",
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.convention = convention
        self.weights = common.jaxify(load_model_data(model_path, convention=convention))

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
    def qpos_joint_names(self) -> list[str]:
        return self.weights.qpos_joint_names

    @property
    def qpos_joint_indices(self) -> list[int]:
        return self.weights.qpos_joint_indices

    @property
    def qpos_joint_axes(self) -> Float[jax.Array, "Q 3"]:
        return self.weights.qpos_joint_axes

    @property
    def qpos_joint_limits(self) -> Float[jax.Array, "Q 2"]:
        return self.weights.qpos_joint_limits

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

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        params = self.get_rest_pose(batch_size=1)
        return self.forward_vertices(
            body_pose=params["body_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )[0]

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_vertices(
        self,
        body_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        body_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    ) -> Float[jax.Array, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            body_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def link_mesh(self, link_name: str) -> dict[str, jax.Array | str | int]:
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

    def joint_meshes(self, joint_name: str) -> list[dict[str, jax.Array | str | int]]:
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

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((batch_size, len(self.weights.qpos_joint_indices), 3), dtype=dtype)
        global_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        body_pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, len(self.weights.qpos_joint_indices)),
            rotation_type=self.rotation_type,
            xp=jnp,
        )
        global_rotation = SO3.identity_as(
            global_ref,
            batch_dims=(batch_size,),
            rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
            xp=jnp,
        )
        return {
            "body_pose": body_pose,
            "global_rotation": global_rotation,
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, value in G1_TPOSE.items():
            if self.rotation_type == "hinge":
                slices = (slice(None), index, 0) if body_pose.ndim == 3 else (slice(None), index)
                body_pose = common.set(body_pose, slices, value, xp=jnp)
            else:
                template = body_pose[:, index, 0, :] if body_pose.ndim == 4 else body_pose[:, index, :]
                axis = jnp.asarray(self.qpos_joint_axes[index], dtype=body_pose.dtype)
                axis_angle = template * 0 + axis * value
                converted = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
                body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, value in G1_APOSE.items():
            if self.rotation_type == "hinge":
                slices = (slice(None), index, 0) if body_pose.ndim == 3 else (slice(None), index)
                body_pose = common.set(body_pose, slices, value, xp=jnp)
            else:
                template = body_pose[:, index, 0, :] if body_pose.ndim == 4 else body_pose[:, index, :]
                axis = jnp.asarray(self.qpos_joint_axes[index], dtype=body_pose.dtype)
                axis_angle = template * 0 + axis * value
                converted = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
                body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, value in G1_IPOSE.items():
            if self.rotation_type == "hinge":
                slices = (slice(None), index, 0) if body_pose.ndim == 3 else (slice(None), index)
                body_pose = common.set(body_pose, slices, value, xp=jnp)
            else:
                template = body_pose[:, index, 0, :] if body_pose.ndim == 4 else body_pose[:, index, :]
                axis = jnp.asarray(self.qpos_joint_axes[index], dtype=body_pose.dtype)
                axis_angle = template * 0 + axis * value
                converted = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
                body_pose = common.set(body_pose, (slice(None), index), converted, xp=jnp)
        params["body_pose"] = body_pose
        return params
