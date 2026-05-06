"""SciPy sparse SMPL backend."""

import numpy as np
from jaxtyping import Float
from scipy import sparse

from body_models.rotations import RotationType
from body_models.smpl.backends import core
from body_models.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplWeights,
    shape: Float[np.ndarray, "B 10"],
    body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
    pelvis_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    v_shaped, j_t, T_world = core.forward_unskinned_vertices(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        posedirs=weights.posedirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=np,
    )
    lbs_weights = sparse.csr_array(weights.lbs_weights)
    if vertex_indices is not None:
        lbs_weights = lbs_weights[vertex_indices]

    R_world = T_world[..., :3, :3]
    t_world = T_world[..., :3, 3]
    t_adjusted = t_world - np.squeeze(R_world @ j_t[..., None], axis=-1)

    v_posed = np.empty_like(v_shaped)
    for batch in range(body_pose.shape[0]):
        W_R = lbs_weights @ R_world[batch].reshape(24, 9)
        W_t = lbs_weights @ t_adjusted[batch]
        v_posed[batch] = np.einsum("vij,vj->vi", W_R.reshape(-1, 3, 3), v_shaped[batch]) + W_t

    return core.apply_global_transform(
        np,
        v_posed,
        global_rotation,
        global_translation,
        rotation_type,
    )


def forward_skeleton(
    weights: SmplWeights,
    shape: Float[np.ndarray, "B 10"],
    body_pose: Float[np.ndarray, "B 23 N"] | Float[np.ndarray, "B 23 3 3"],
    pelvis_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return core.forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=np,
    )
