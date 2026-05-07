"""NumPy FLAME backend."""

import numpy as np
from jaxtyping import Float

from body_models.flame.backends.core import forward_skeleton as _forward_skeleton
from body_models.flame.backends.core import forward_vertices as _forward_vertices
from body_models.flame.io import FlameWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: FlameWeights,
    shape: Float[np.ndarray, "B S"],
    expression: Float[np.ndarray, "B E"],
    pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
    head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_vertices(
        v_template=weights.v_template,
        v_template_full=weights.v_template_full,
        shapedirs=weights.shapedirs,
        shapedirs_full=weights.shapedirs_full,
        exprdirs=weights.exprdirs,
        exprdirs_full=weights.exprdirs_full,
        posedirs=weights.posedirs,
        lbs_weights=weights.lbs_weights,
        J_regressor=weights.J_regressor,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        expression=expression,
        pose=pose,
        head_rotation=head_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        xp=np,
    )


def forward_skeleton(
    weights: FlameWeights,
    shape: Float[np.ndarray, "B S"],
    expression: Float[np.ndarray, "B E"],
    pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
    head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return _forward_skeleton(
        v_template_full=weights.v_template_full,
        shapedirs_full=weights.shapedirs_full,
        exprdirs_full=weights.exprdirs_full,
        J_regressor=weights.J_regressor,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        expression=expression,
        pose=pose,
        head_rotation=head_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=np,
    )
