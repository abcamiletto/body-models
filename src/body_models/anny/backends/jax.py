"""JAX ANNY backend."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from body_models.anny.backends.core import AnnyIdentity
from body_models.anny.backends.core import forward_skeleton as _forward_skeleton
from body_models.anny.backends.core import forward_vertices as _forward_vertices
from body_models.anny.backends.core import prepare_identity as _prepare_identity
from body_models.anny.io import AnnyWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def prepare_identity(
    weights: AnnyWeights,
    shape: Float[jax.Array, "*batch 6"],
    extrapolate_phenotypes: bool = False,
    skip_vertices: bool = False,
) -> AnnyIdentity:
    return _prepare_identity(
        xp=jnp,
        template_vertices=weights.template_vertices,
        blendshapes=weights.blendshapes,
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: AnnyWeights,
    pose: Float[jax.Array, "*batch J N"] | Float[jax.Array, "*batch J 3 3"],
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[jax.Array, "*batch J 4 4"],
    rest_vertices: Float[jax.Array, "*batch V 3"],
):
    return _forward_vertices(
        template_vertices=weights.template_vertices,
        blendshapes=weights.blendshapes,
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        lbs_weights=weights.lbs_weights,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        kinematic_fronts=weights.kinematic_fronts,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_bone_poses=rest_bone_poses,
        rest_vertices=rest_vertices,
        xp=jnp,
    )


def forward_skeleton(
    weights: AnnyWeights,
    pose: Float[jax.Array, "*batch J N"] | Float[jax.Array, "*batch J 3 3"],
    global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    global_translation: Float[jax.Array, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[jax.Array, "*batch J 4 4"],
    rest_vertices: Float[jax.Array, "*batch V 3"] | None = None,
):
    return _forward_skeleton(
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        kinematic_fronts=weights.kinematic_fronts,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=extrapolate_phenotypes,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        rest_bone_poses=rest_bone_poses,
        rest_vertices=rest_vertices,
        xp=jnp,
    )
