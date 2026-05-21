"""NumPy GarmentMeasurements backend."""

import numpy as np
from jaxtyping import Float

from body_models.garment_measurements.backends.core import (
    GarmentMeasurementsIdentity,
    GarmentMeasurementsPreparedPose,
)
from body_models.garment_measurements.backends.core import forward_skeleton as _forward_skeleton
from body_models.garment_measurements.backends.core import forward_vertices as _forward_vertices
from body_models.garment_measurements.backends.core import prepare_identity as _prepare_identity
from body_models.garment_measurements.backends.core import prepare_pose as _prepare_pose
from body_models.garment_measurements.io import GarmentMeasurementsWeights
from body_models.rotations import RotationType

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: GarmentMeasurementsWeights,
    shape: Float[np.ndarray, "*batch C"],
    skip_vertices: bool = False,
) -> GarmentMeasurementsIdentity:
    return _prepare_identity(
        xp=np,
        mean_vertices=weights.mean_vertices,
        components=weights.components,
        eigenvalues=weights.eigenvalues,
        bind_quats=weights.bind_quats,
        mvc_weights=weights.mvc_weights,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        skip_vertices=skip_vertices,
    )


def prepare_pose(
    weights: GarmentMeasurementsWeights,
    pose: Float[np.ndarray, "*batch J N"] | Float[np.ndarray, "*batch J 3 3"],
    rotation_type: RotationType = "axis_angle",
    *,
    bind_skeleton: Float[np.ndarray, "*batch J 7"],
    local_bind_translations: Float[np.ndarray, "*batch J 3"],
    skip_vertices: bool = False,
) -> GarmentMeasurementsPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return _prepare_pose(
        xp=np,
        bind_quats=weights.bind_quats,
        kinematic_fronts=weights.kinematic_fronts,
        pose=pose,
        rotation_type=rotation_type,
        bind_skeleton=bind_skeleton,
        local_bind_translations=local_bind_translations,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: GarmentMeasurementsWeights,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    rest_vertices: Float[np.ndarray, "*batch V 3"],
    skinning_skeleton: Float[np.ndarray, "*batch J 7"],
):
    return _forward_vertices(
        skin_weights=weights.skin_weights,
        global_rotation=global_rotation,
        global_translation=global_translation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
        rest_vertices=rest_vertices,
        skinning_skeleton=skinning_skeleton,
        xp=np,
    )


def forward_skeleton(
    weights: GarmentMeasurementsWeights,
    global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    posed_skeleton: Float[np.ndarray, "*batch J 7"],
):
    return _forward_skeleton(
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        posed_skeleton=posed_skeleton,
        xp=np,
    )
