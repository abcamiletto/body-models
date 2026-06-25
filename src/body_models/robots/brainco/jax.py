"""JAX backend for the BrainCo Revo 2 robotic hand model."""

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.base import MeshPayload, RigidBodyModel
from body_models.robots.brainco.backends import core
from body_models.robots.brainco.backends import jax as backend
from body_models.robots.brainco.io import Side, load_model_data
from body_models.robots.brainco.constants import BRAINCO_HAND_PRESETS, LEFT_BRAINCO_JOINTS, RIGHT_BRAINCO_JOINTS

__all__ = ["BrainCoHand"]


class BrainCoHand(RigidBodyModel):
    """BrainCo Revo 2 as rigid STL links attached to its MuJoCo hand skeleton."""

    has_hands = True

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        side: Side = "right",
        rotation_type: core.RotationType = "rotmat",
    ) -> None:
        """Initialize the BrainCoHand model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            side: Hand side to load.
            rotation_type: Rotation representation expected by pose inputs.
        """
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.weights = common.jaxify(load_model_data(model_path, side=side))

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
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
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    def forward_skeleton(
        self,
        hand_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            hand_pose: Local hand joint rotations.
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
            rotation_type=self.rotation_type,
        )

    def forward_meshes(
        self,
        hand_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        link_indices: list[int] | None = None,
    ) -> list[MeshPayload]:
        """Compute posed link meshes.

        Args:
            hand_pose: Local hand joint rotations.
            global_translation: Global model translation.
            global_rotation: Global model rotation.
            link_indices: Optional subset of links to return.

        Returns:
            One posed mesh payload per link.
        """
        return backend.forward_meshes(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            link_indices=link_indices,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        hand_pose: Float[jax.Array, "B Q N"] | Float[jax.Array, "B Q 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    ) -> Float[jax.Array, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def link_mesh(self, link_name: str) -> MeshPayload:
        return core.link_mesh(
            self.weights.vertices,
            self.weights.faces,
            self.weights.link_joint_indices,
            self.weights.link_vertex_starts,
            self.weights.link_vertex_counts,
            self.weights.link_face_starts,
            self.weights.link_face_counts,
            self.weights.joint_names,
            self.weights.link_names,
            link_name,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=jnp.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, jax.Array]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        global_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        qpos = jnp.zeros((len(self.weights.qpos_joint_indices), 1), dtype=dtype)
        if hands != "default":
            qpos = jnp.asarray(BRAINCO_HAND_PRESETS[self.side][hands], dtype=dtype).reshape(-1, 1)
        qpos = jnp.broadcast_to(qpos, (*batch_dims, *qpos.shape))
        axes = self.weights.qpos_joint_axes
        rotmat = SO3.convert(qpos, src="hinge", dst="rotmat", src_kwargs={"axes": axes}, xp=jnp)
        dst_kwargs = {"hinge": {"axes": axes}}.get(self.rotation_type, {})
        hand_pose = SO3.convert(
            rotmat,
            src="rotmat",
            dst=self.rotation_type,
            dst_kwargs=dst_kwargs,
            xp=jnp,
        )
        return {
            "hand_pose": hand_pose,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=jnp,
            ),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }
