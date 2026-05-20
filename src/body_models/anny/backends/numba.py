"""Numba ANNY backend."""

import numpy as np
from jaxtyping import Float
from numba import njit, prange

from body_models.anny.backends import core
from body_models.anny.io import AnnyWeights
from body_models.rotations import RotationType


def prepare_identity(
    weights: AnnyWeights,
    shape: Float[np.ndarray, "*batch 6"],
    extrapolate_phenotypes: bool = False,
    skip_vertices: bool = False,
) -> core.AnnyIdentity:
    return core.prepare_identity(
        xp=np,
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


__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def forward_vertices(
    weights: AnnyWeights,
    pose: Float[np.ndarray, "*batch J N"] | Float[np.ndarray, "*batch J 3 3"],
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[np.ndarray, "*batch J 4 4"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
):
    rest_verts, bone_transforms = core.forward_unskinned_vertices(
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
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_bone_poses=rest_bone_poses,
        rest_vertices=rest_vertices,
        xp=np,
    )

    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    vertices = np.empty_like(rest_verts)
    for batch in np.ndindex(rest_verts.shape[:-2]):
        vertices[batch] = _skin_vertices(
            rest_verts[batch][None],
            bone_transforms[batch][None, :, :3, :3],
            bone_transforms[batch][None, :, :3, 3],
            joint_indices,
            joint_weights,
        )[0]
    return core.apply_global_transform(np, vertices, global_rotation, global_translation, rotation_type)


def forward_skeleton(
    weights: AnnyWeights,
    pose: Float[np.ndarray, "*batch J N"] | Float[np.ndarray, "*batch J 3 3"],
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    extrapolate_phenotypes: bool = False,
    *,
    rest_bone_poses: Float[np.ndarray, "*batch J 4 4"],
    rest_vertices: Float[np.ndarray, "*batch V 3"] | None = None,
):
    return core.forward_skeleton(
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
        xp=np,
    )


@njit(parallel=True, fastmath=True)
def _skin_vertices(
    rest_verts: Float[np.ndarray, "B V 3"],
    R: Float[np.ndarray, "B J 3 3"],
    t: Float[np.ndarray, "B J 3"],
    joint_indices,
    joint_weights,
) -> Float[np.ndarray, "B V 3"]:
    batch_size, num_vertices = rest_verts.shape[:2]
    output = np.empty_like(rest_verts)

    for batch in prange(batch_size):  # ty: ignore[not-iterable]
        for vertex in range(num_vertices):
            vx = rest_verts[batch, vertex, 0]
            vy = rest_verts[batch, vertex, 1]
            vz = rest_verts[batch, vertex, 2]
            out_x = 0.0
            out_y = 0.0
            out_z = 0.0

            for slot in range(joint_indices.shape[1]):
                joint = joint_indices[vertex, slot]
                weight = joint_weights[vertex, slot]

                out_x += weight * (
                    R[batch, joint, 0, 0] * vx
                    + R[batch, joint, 0, 1] * vy
                    + R[batch, joint, 0, 2] * vz
                    + t[batch, joint, 0]
                )
                out_y += weight * (
                    R[batch, joint, 1, 0] * vx
                    + R[batch, joint, 1, 1] * vy
                    + R[batch, joint, 1, 2] * vz
                    + t[batch, joint, 1]
                )
                out_z += weight * (
                    R[batch, joint, 2, 0] * vx
                    + R[batch, joint, 2, 1] * vy
                    + R[batch, joint, 2, 2] * vz
                    + t[batch, joint, 2]
                )

            output[batch, vertex, 0] = out_x
            output[batch, vertex, 1] = out_y
            output[batch, vertex, 2] = out_z

    return output
