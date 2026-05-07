"""NumPy MANO backend."""

import numpy as np
from jaxtyping import Float

from body_models.mano.backends.core import forward_skeleton as _forward_skeleton
from body_models.mano.backends.core import forward_vertices as _forward_vertices
from body_models.mano.io import ManoWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: ManoWeights,
    shape: Float[np.ndarray, "B 10"],
    hand_pose: Float[np.ndarray, "B 15 N"] | Float[np.ndarray, "B 15 3 3"],
    wrist_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        posedirs=weights.posedirs,
        lbs_weights=weights.lbs_weights,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=np,
    )


def forward_skeleton(
    weights: ManoWeights,
    shape: Float[np.ndarray, "B 10"],
    hand_pose: Float[np.ndarray, "B 15 N"] | Float[np.ndarray, "B 15 3 3"],
    wrist_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        hand_mean=weights.hand_mean,
        shape=shape,
        hand_pose=hand_pose,
        wrist_rotation=wrist_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=np,
    )
