"""Numba GarmentMeasurements backend."""

import numpy as np
from jaxtyping import Float
from nanomanifold import SE3
from numba import njit, prange

from body_models.garment_measurements.backends import core
from body_models.garment_measurements.io import GarmentMeasurementsWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: GarmentMeasurementsWeights,
    shape: Float[np.ndarray, "B C"],
    pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"],
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    vertices, skeleton = core.forward_unskinned_vertices(
        mean_vertices=weights.mean_vertices,
        components=weights.components,
        eigenvalues=weights.eigenvalues,
        bind_quats=weights.bind_quats,
        mvc_weights=weights.mvc_weights,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        pose=pose,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=np,
    )
    joint_indices = weights.skin_joint_indices
    joint_weights = weights.skin_joint_weights
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    skin_mats = SE3.to_matrix(skeleton, xp=np)
    skinned = np.empty_like(vertices)
    for batch in np.ndindex(vertices.shape[:-2]):
        skinned[batch] = _skin_vertices(
            vertices[batch][None],
            skin_mats[batch][None, :, :3, :3],
            skin_mats[batch][None, :, :3, 3],
            joint_indices,
            joint_weights,
        )[0]
    return core._apply_global_transform(
        values=skinned,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
        xp=np,
    )


def forward_skeleton(
    weights: GarmentMeasurementsWeights,
    shape: Float[np.ndarray, "B C"],
    pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"],
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return core.forward_skeleton(
        mean_vertices=weights.mean_vertices,
        components=weights.components,
        eigenvalues=weights.eigenvalues,
        bind_quats=weights.bind_quats,
        mvc_weights=weights.mvc_weights,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        pose=pose,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=np,
    )


@njit(parallel=True, fastmath=True)
def _skin_vertices(vertices, R, t, joint_indices, joint_weights):
    batch_size, num_vertices = vertices.shape[:2]
    output = np.empty_like(vertices)

    for batch in prange(batch_size):  # ty: ignore[not-iterable]
        for vertex in range(num_vertices):
            vx = vertices[batch, vertex, 0]
            vy = vertices[batch, vertex, 1]
            vz = vertices[batch, vertex, 2]
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
