"""JAX SMPL-X backend."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.rotations import RotationType
from body_models.smplx.backends.core import forward_skeleton as _forward_skeleton
from body_models.smplx.backends.core import forward_vertices as _forward_vertices
from body_models.smplx.io import SmplxWeights

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplxWeights,
    shape: Float[jax.Array, "B 10"],
    body_pose: Float[jax.Array, "B 21 N"] | Float[jax.Array, "B 21 3 3"],
    hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
    head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
    expression: Float[jax.Array, "B 10"] | None = None,
    pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        exprdirs=weights.exprdirs,
        posedirs=weights.posedirs,
        lbs_weights=weights.lbs_weights,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        expression=expression,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=jnp,
    )


def forward_skeleton(
    weights: SmplxWeights,
    shape: Float[jax.Array, "B 10"],
    body_pose: Float[jax.Array, "B 21 N"] | Float[jax.Array, "B 21 3 3"],
    hand_pose: Float[jax.Array, "B 30 N"] | Float[jax.Array, "B 30 3 3"],
    head_pose: Float[jax.Array, "B 3 N"] | Float[jax.Array, "B 3 3 3"],
    expression: Float[jax.Array, "B 10"] | None = None,
    pelvis_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        body_pose=body_pose,
        hand_pose=hand_pose,
        head_pose=head_pose,
        expression=expression,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=jnp,
    )
