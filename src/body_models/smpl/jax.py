"""JAX backend for SMPL model."""

from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smpl.backends.core import SmplIdentity, SmplPreparedPose
from body_models.smpl.backends import jax as backend
from body_models.smpl.constants import SMPL_BODY_PRESETS, SMPL_JOINT_NAMES, SMPL_JOINTS
from body_models.smpl.io import get_model_path, load_model_data


__all__ = ["SMPL"]


@jax.tree_util.register_pytree_node_class
class SMPL(BodyModel):
    """SMPL body model with JAX backend."""

    NUM_BODY_JOINTS = 23
    NUM_JOINTS = 24
    JOINTS = SMPL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
    ):
        """Initialize the SMPL model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
        """
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")

        # Default gender to "neutral" for attribute storage when model_path is given
        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1

        resolved_path = get_model_path(model_path, gender)
        data = load_model_data(resolved_path, simplify=simplify)

        self.weights = common.jaxify(data)

    def tree_flatten(self):
        children = (self.weights,)
        aux_data = (
            self.gender,
            self.rotation_type,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.weights,) = children
        gender, rotation_type = aux_data
        obj.gender = gender
        obj.rotation_type = rotation_type
        return obj

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(SMPL_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[jax.Array, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[jax.Array, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[jax.Array, "V 24"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        body_pose: Float[jax.Array, "*batch 23 N"] | Float[jax.Array, "*batch 23 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: SmplIdentity | None = None,
    ) -> Float[jax.Array, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)
        pose = self.prepare_pose(body_pose, pelvis_rotation, identity=identity)
        return backend.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            **identity,
            **pose,
        )

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "*batch 23 N"] | Float[jax.Array, "*batch 23 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        global_translation: Float[jax.Array, "*batch 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[jax.Array, "*batch 10"] | None = None,
        identity: SmplIdentity | None = None,
    ) -> Float[jax.Array, "*batch 24 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            batch_shape = body_pose.shape[: -(self.num_rot_dims + 1)]
            shape = jnp.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        pose = self.prepare_pose(body_pose, pelvis_rotation, identity=identity, skip_vertices=True)
        return backend.forward_skeleton(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            **identity,
            **pose,
        )

    def prepare_identity(
        self,
        shape: Float[jax.Array, "*batch 10"],
        expression: Any | None = None,
        skip_vertices: bool = False,
    ) -> SmplIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        if expression is not None:
            raise ValueError("SMPL does not support expression parameters.")
        return backend.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        body_pose: Float[jax.Array, "*batch 23 N"] | Float[jax.Array, "*batch 23 3 3"],
        pelvis_rotation: Float[jax.Array, "*batch N"] | Float[jax.Array, "*batch 3 3"] | None = None,
        *,
        identity: SmplIdentity,
        skip_vertices: bool = False,
    ) -> SmplPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return backend.prepare_pose(
            self.weights,
            body_pose,
            pelvis_rotation,
            rotation_type=self.rotation_type,
            local_joint_offsets=identity["local_joint_offsets"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=jnp.float32) -> dict[str, jax.Array]:
        body_pose_ref = jnp.zeros((*batch_dims, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        pelvis_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        return {
            "shape": jnp.zeros((*batch_dims, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(*batch_dims, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
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
        axis_angle = jnp.asarray(SMPL_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        axis_angle = jnp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
        return params
