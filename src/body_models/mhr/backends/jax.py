"""JAX MHR backend."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.mhr.backends.core import MhrIdentity
from body_models.mhr.backends.core import forward_skeleton as _forward_skeleton
from body_models.mhr.backends.core import forward_vertices as _forward_vertices
from body_models.mhr.backends.core import prepare_identity as _prepare_identity
from body_models.mhr.io import MhrWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def prepare_identity(
    weights: MhrWeights,
    shape: Float[jax.Array, "*batch 45"],
    expression: Float[jax.Array, "*batch 72"] | None = None,
    skip_vertices: bool = False,
) -> MhrIdentity:
    return _prepare_identity(
        xp=jnp,
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        shape=shape,
        expression=expression,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: MhrWeights,
    pose: Float[jax.Array, "*batch 204"],
    global_rotation: Float[jax.Array, "B 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    *,
    rest_vertices: Float[jax.Array, "*batch V 3"],
):
    return _forward_vertices(
        base_vertices=weights.base_vertices,
        blendshape_dirs=weights.blendshape_dirs,
        skin_weights=weights.skin_weights,
        skin_indices=weights.skin_indices,
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        bind_inv_linear=weights.bind_inv_linear,
        bind_inv_translation=weights.bind_inv_translation,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        expr_dim=72,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        corrective_W1=weights.corrective_W1,
        corrective_W2=weights.corrective_W2,
        rest_vertices=rest_vertices,
        xp=jnp,
    )


def forward_skeleton(
    weights: MhrWeights,
    pose: Float[jax.Array, "*batch 204"],
    global_rotation: Float[jax.Array, "B 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    *,
    rest_vertices: Float[jax.Array, "*batch V 3"] | None = None,
):
    return _forward_skeleton(
        joint_offsets=weights.joint_offsets,
        joint_pre_rotations=weights.joint_pre_rotations,
        parameter_transform=weights.parameter_transform,
        kinematic_fronts=weights.kinematic_fronts,
        num_joints=len(weights.parents),
        shape_dim=45,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        xp=jnp,
    )
