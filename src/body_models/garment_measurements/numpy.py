"""NumPy backend for the GarmentMeasurements PCA body model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES, RotationType
from .backends import numpy as numpy_backend
from .io import get_model_path, load_model_data


__all__ = ["GarmentMeasurements"]

Backend = Literal["numpy", "numba"]
FLAVORS = ("numpy", "numba")


class GarmentMeasurements(BodyModel):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    flavors = FLAVORS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
        backend: Backend = "numpy",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if backend not in FLAVORS:
            raise ValueError(f"Invalid backend: {backend}")

        self.weights = load_model_data(get_model_path(model_path), dtype=np.float32)
        self.rotation_type = rotation_type
        self._kernel = _get_kernel(backend)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.weights.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.mean_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B C"],
        pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B C"],
        pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        pose_ref = np.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        global_ref = np.zeros((batch_size,), dtype=dtype)
        return {
            "shape": np.zeros((1, self.num_shape_components), dtype=dtype),
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }


def _get_kernel(backend: Backend):
    if backend == "numpy":
        return numpy_backend

    try:
        from body_models.garment_measurements.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use GarmentMeasurements backend='numba'.") from exc

    return numba_backend
