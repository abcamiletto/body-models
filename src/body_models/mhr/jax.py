"""JAX backend for MHR model."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from body_models.mhr.backends import jax as backend
from body_models.mhr.io import MhrWeights, get_model_path, load_model_data

__all__ = ["MHR"]


def _flatten_mhr_weights(weights: MhrWeights):
    children = (
        weights.base_vertices,
        weights.blendshape_dirs,
        weights.skin_weights,
        weights.skin_indices,
        weights.faces,
        weights.joint_offsets,
        weights.joint_pre_rotations,
        weights.parameter_transform,
        weights.bind_inv_linear,
        weights.bind_inv_translation,
        weights.corrective_W1,
        weights.corrective_W2,
    )
    aux_data = (
        tuple(weights.parents),
        tuple((tuple(joints), tuple(parents)) for joints, parents in weights.kinematic_fronts),
        tuple(weights.joint_names),
    )
    return children, aux_data


def _unflatten_mhr_weights(aux_data, children):
    (
        base_vertices,
        blendshape_dirs,
        skin_weights,
        skin_indices,
        faces,
        joint_offsets,
        joint_pre_rotations,
        parameter_transform,
        bind_inv_linear,
        bind_inv_translation,
        corrective_W1,
        corrective_W2,
    ) = children
    parents, kinematic_fronts, joint_names = aux_data
    return MhrWeights(
        base_vertices=base_vertices,
        blendshape_dirs=blendshape_dirs,
        skin_weights=skin_weights,
        skin_indices=skin_indices,
        faces=faces,
        joint_offsets=joint_offsets,
        joint_pre_rotations=joint_pre_rotations,
        parameter_transform=parameter_transform,
        bind_inv_linear=bind_inv_linear,
        bind_inv_translation=bind_inv_translation,
        corrective_W1=corrective_W1,
        corrective_W2=corrective_W2,
        parents=list(parents),
        kinematic_fronts=[(list(joints), list(parents)) for joints, parents in kinematic_fronts],
        joint_names=list(joint_names),
    )


jax.tree_util.register_pytree_node(MhrWeights, _flatten_mhr_weights, _unflatten_mhr_weights)


@jax.tree_util.register_pytree_node_class
class MHR(BodyModel):
    """MHR body model with JAX backend."""

    SHAPE_DIM = 45
    EXPR_DIM = 72

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        self.weights = common.jaxify(load_model_data(get_model_path(model_path), lod=lod, simplify=simplify))

    def tree_flatten(self):
        return (self.weights,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.weights,) = children
        return obj

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        dense = jnp.zeros((self.weights.skin_weights.shape[0], self.num_joints), dtype=self.weights.skin_weights.dtype)
        return dense.at[jnp.arange(self.weights.skin_weights.shape[0])[:, None], self.weights.skin_indices].set(
            self.weights.skin_weights
        )

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 45"],
        pose: Float[jax.Array, "B 204"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 45"],
        pose: Float[jax.Array, "B 204"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[jax.Array, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": jnp.zeros((batch_size, self.pose_dim), dtype=dtype),
            "expression": jnp.zeros((batch_size, self.EXPR_DIM), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
