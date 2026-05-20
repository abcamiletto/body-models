"""NumPy FLAME backend."""

import numpy as np
from jaxtyping import Float

from body_models.flame.backends.core import FlameIdentity
from body_models.flame.backends.core import forward_skeleton as _forward_skeleton
from body_models.flame.backends.core import forward_vertices as _forward_vertices
from body_models.flame.backends.core import prepare_identity as _prepare_identity
from body_models.flame.io import FlameWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity"]


def prepare_identity(
    weights: FlameWeights,
    shape: Float[np.ndarray, "*batch S"],
    expression: Float[np.ndarray, "*batch E"],
    skip_vertices: bool = False,
) -> FlameIdentity:
    return _prepare_identity(
        xp=np,
        v_template=weights.v_template,
        shapedirs=weights.shapedirs,
        exprdirs=weights.exprdirs,
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        shape=shape,
        expression=expression,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: FlameWeights,
    pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
    head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"],
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
        pose=pose,
        head_rotation=head_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        rest_vertices=rest_vertices,
        xp=np,
    )


def forward_skeleton(
    weights: FlameWeights,
    pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
    head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_joints: Float[np.ndarray, "*batch J 3"],
    local_joint_offsets: Float[np.ndarray, "*batch J 3"],
    rest_vertices: Float[np.ndarray, "*batch V 3"] | None = None,
):
    return _forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        j_exprdirs=weights.j_exprdirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        pose=pose,
        head_rotation=head_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        rest_joints=rest_joints,
        local_joint_offsets=local_joint_offsets,
        rest_vertices=rest_vertices,
        xp=np,
    )
