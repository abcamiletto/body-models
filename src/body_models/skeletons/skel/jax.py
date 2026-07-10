"""JAX backend for SKEL model."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import SkinnedModel
from body_models.skeletons.skel.backends import jax as backend
from body_models.skeletons.skel.backends.core import SkelIdentity, SkelPreparedPose
from body_models.skeletons.skel.constants import SKEL_BODY_PRESETS, SKEL_JOINTS
from body_models.skeletons.skel.io import get_model_path, load_model_data
from body_models.skeletons.skel.pose import (
    SKEL_BODY_POSE_DIM,
    SKEL_CANONICAL_POSE_DIM,
    SKEL_HEAD_POSE_DIM,
    pack_pose,
    unpack_pose,
)

__all__ = ["SKEL"]


class SKEL(SkinnedModel):
    """SKEL body model with JAX backend."""

    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose")
    NUM_BETAS = 10
    NUM_JOINTS = 24
    JOINTS = SKEL_JOINTS
    has_head = True

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ):
        """Initialize the SKEL model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            simplify: Mesh simplification factor to apply while loading.
        """
        if gender not in {"male", "female"}:
            raise ValueError(f"Invalid gender: {gender}. Must be 'male' or 'female'.")
        assert simplify >= 1.0

        self.gender = gender
        data = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        self.weights = common.jaxify(data)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def pose_dim(self) -> int:
        return SKEL_CANONICAL_POSE_DIM

    @property
    def body_pose_dim(self) -> int:
        return SKEL_BODY_POSE_DIM

    @property
    def head_pose_dim(self) -> int:
        return SKEL_HEAD_POSE_DIM

    def forward_vertices(
        self,
        body_pose: Float[jax.Array, "*batch 43"],
        head_pose: Float[jax.Array, "*batch 3"],
        global_rotation: Float[jax.Array, "*batch 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: SkelIdentity | None = None,
    ) -> Float[jax.Array, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head joint rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)
        pose = self.prepare_pose(body_pose, head_pose, identity=identity)
        return backend.forward_vertices(
            self.weights,
            identity["rest_vertices"],
            pose["skinning_transforms"],
            pose["pose_offsets"],
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "*batch 43"],
        head_pose: Float[jax.Array, "*batch 3"],
        global_rotation: Float[jax.Array, "*batch 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: SkelIdentity | None = None,
    ) -> Float[jax.Array, "*batch 24 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head joint rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[:-1]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        pose = self.prepare_pose(body_pose, head_pose, identity=identity, skip_vertices=True)
        return backend.forward_skeleton(
            self.weights,
            pose["skeleton_transforms"],
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def prepare_identity(
        self,
        shape: Float[jax.Array, "*batch 10"],
        skip_vertices: bool = False,
    ) -> SkelIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return backend.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[jax.Array, "*batch 43"],
        head_pose: Float[jax.Array, "*batch 3"],
        *,
        identity: SkelIdentity,
        skip_vertices: bool = False,
    ) -> SkelPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose = pack_pose(jnp, body_pose, head_pose)
        return backend.prepare_pose(
            self.weights,
            pose,
            local_joint_offsets=identity["local_joint_offsets"],
            rest_joints=identity["rest_joints"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((*batch_dims, self.NUM_BETAS), dtype=dtype),
            "body_pose": jnp.zeros((*batch_dims, self.body_pose_dim), dtype=dtype),
            "head_pose": jnp.zeros((*batch_dims, self.head_pose_dim), dtype=dtype),
            "global_rotation": jnp.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, jax.Array]:
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        pose = jnp.asarray(SKEL_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        pose = jnp.broadcast_to(pose, (*batch_dims, *pose.shape))
        params["body_pose"], params["head_pose"] = unpack_pose(jnp, pose)
        return params
