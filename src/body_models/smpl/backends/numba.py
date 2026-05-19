"""Numba SMPL backend."""

import numpy as np
from jaxtyping import Float, Int
from numba import njit, prange

from body_models.rotations import RotationType
from body_models.smpl.backends import core
from body_models.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def prepare_identity(
    weights: SmplWeights,
    shape: Float[np.ndarray, "*batch 10"],
    skip_vertices: bool = False,
) -> core.SmplIdentity:
    return core.prepare_identity(
        xp=np,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def numba_skin(
    v_shaped,
    j_t,
    T_world,
    joint_indices,
    joint_weights,
    *,
    global_rotation=None,
    global_translation=None,
    rotation_type: RotationType = "axis_angle",
):
    v_posed = np.empty_like(v_shaped)
    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    for batch in np.ndindex(v_shaped.shape[:-2]):
        v_posed[batch] = _skin_vertices(
            v_shaped[batch][None],
            j_t[batch][None],
            R_world[batch][None],
            t_world[batch][None],
            joint_indices,
            joint_weights,
        )[0]
    return core.apply_global_transform(np, v_posed, global_rotation, global_translation, rotation_type)


def forward_vertices(
    weights: SmplWeights,
    body_pose: Float[np.ndarray, "*batch 23 N"] | Float[np.ndarray, "*batch 23 3 3"],
    pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
):
    v_shaped, j_t, T_world = core.forward_unskinned_vertices(
        posedirs=weights.posedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        rest_vertices=rest_vertices,
        xp=np,
    )

    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    return numba_skin(
        v_shaped,
        j_t,
        T_world,
        joint_indices,
        joint_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        rotation_type=rotation_type,
    )


def forward_skeleton(
    weights: SmplWeights,
    body_pose: Float[np.ndarray, "*batch 23 N"] | Float[np.ndarray, "*batch 23 3 3"],
    pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
    global_translation: Float[np.ndarray, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"] | None = None,
):
    return core.forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        xp=np,
    )


@njit(parallel=True, fastmath=True)
def _skin_vertices(
    v_shaped: Float[np.ndarray, "B V 3"],
    j_t: Float[np.ndarray, "B J 3"],
    R_world: Float[np.ndarray, "B J 3 3"],
    t_world: Float[np.ndarray, "B J 3"],
    joint_indices: Int[np.ndarray, "V K"],
    joint_weights: Float[np.ndarray, "V K"],
) -> Float[np.ndarray, "B V 3"]:
    batch_size, num_vertices = v_shaped.shape[:2]
    output = np.empty_like(v_shaped)

    for batch in prange(batch_size):  # ty: ignore[not-iterable]
        for vertex in range(num_vertices):
            vx = v_shaped[batch, vertex, 0]
            vy = v_shaped[batch, vertex, 1]
            vz = v_shaped[batch, vertex, 2]
            out_x = 0.0
            out_y = 0.0
            out_z = 0.0

            for slot in range(joint_indices.shape[1]):
                joint = joint_indices[vertex, slot]
                if joint < 0:
                    continue

                weight = joint_weights[vertex, slot]
                jx = j_t[batch, joint, 0]
                jy = j_t[batch, joint, 1]
                jz = j_t[batch, joint, 2]

                r00 = R_world[batch, joint, 0, 0]
                r01 = R_world[batch, joint, 0, 1]
                r02 = R_world[batch, joint, 0, 2]
                r10 = R_world[batch, joint, 1, 0]
                r11 = R_world[batch, joint, 1, 1]
                r12 = R_world[batch, joint, 1, 2]
                r20 = R_world[batch, joint, 2, 0]
                r21 = R_world[batch, joint, 2, 1]
                r22 = R_world[batch, joint, 2, 2]

                out_x += weight * (r00 * (vx - jx) + r01 * (vy - jy) + r02 * (vz - jz) + t_world[batch, joint, 0])
                out_y += weight * (r10 * (vx - jx) + r11 * (vy - jy) + r12 * (vz - jz) + t_world[batch, joint, 1])
                out_z += weight * (r20 * (vx - jx) + r21 * (vy - jy) + r22 * (vz - jz) + t_world[batch, joint, 2])

            output[batch, vertex, 0] = out_x
            output[batch, vertex, 1] = out_y
            output[batch, vertex, 2] = out_z

    return output


@njit(parallel=True, fastmath=True)
def _forward_kinematics(
    R_local: Float[np.ndarray, "B J 3 3"],
    t_local: Float[np.ndarray, "B J 3"],
    parents: Int[np.ndarray, "J"],
) -> tuple[Float[np.ndarray, "B J 3 3"], Float[np.ndarray, "B J 3"]]:
    batch_size, num_joints = t_local.shape[:2]
    R_world = np.empty_like(R_local)
    t_world = np.empty_like(t_local)

    for batch in prange(batch_size):  # ty: ignore[not-iterable]
        for joint in range(num_joints):
            parent = parents[joint]
            if parent < 0:
                for row in range(3):
                    t_world[batch, joint, row] = t_local[batch, joint, row]
                    for col in range(3):
                        R_world[batch, joint, row, col] = R_local[batch, joint, row, col]
                continue

            for row in range(3):
                t = t_world[batch, parent, row]
                for k in range(3):
                    t += R_world[batch, parent, row, k] * t_local[batch, joint, k]
                t_world[batch, joint, row] = t

                for col in range(3):
                    R = 0.0
                    for k in range(3):
                        R += R_world[batch, parent, row, k] * R_local[batch, joint, k, col]
                    R_world[batch, joint, row, col] = R

    return R_world, t_world


@njit(fastmath=True)
def _forward_kinematics_matrix(
    R_local: Float[np.ndarray, "B J 3 3"],
    t_local: Float[np.ndarray, "B J 3"],
    parents: Int[np.ndarray, "J"],
    global_translation: Float[np.ndarray, "B 3"],
) -> Float[np.ndarray, "B J 4 4"]:
    batch_size, num_joints = t_local.shape[:2]
    R_world = np.empty_like(R_local)
    t_world = np.empty_like(t_local)
    T_world = np.zeros((batch_size, num_joints, 4, 4), dtype=t_local.dtype)

    for batch in range(batch_size):
        for joint in range(num_joints):
            parent = parents[joint]
            if parent < 0:
                for row in range(3):
                    t_world[batch, joint, row] = t_local[batch, joint, row]
                    for col in range(3):
                        R_world[batch, joint, row, col] = R_local[batch, joint, row, col]
            else:
                for row in range(3):
                    t = t_world[batch, parent, row]
                    for k in range(3):
                        t += R_world[batch, parent, row, k] * t_local[batch, joint, k]
                    t_world[batch, joint, row] = t

                    for col in range(3):
                        R = 0.0
                        for k in range(3):
                            R += R_world[batch, parent, row, k] * R_local[batch, joint, k, col]
                        R_world[batch, joint, row, col] = R

            for row in range(3):
                T_world[batch, joint, row, 3] = t_world[batch, joint, row] + global_translation[batch, row]
                for col in range(3):
                    T_world[batch, joint, row, col] = R_world[batch, joint, row, col]
            T_world[batch, joint, 3, 3] = 1.0

    return T_world
