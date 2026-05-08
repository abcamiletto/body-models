"""NumPy backend for FLAME model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.flame.backends import numpy as numpy_backend
from body_models.flame.backends import scipy as scipy_backend
from body_models.flame.constants import FLAME_JOINT_NAMES
from body_models.flame.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

Backend = Literal["numpy", "scipy", "numba"]
FLAVORS = ("numpy", "scipy", "numba")

__all__ = ["FLAME"]


class FLAME(BodyModel):
    """FLAME head model with NumPy backend."""

    NUM_HEAD_JOINTS = 4
    NUM_JOINTS = 5
    flavors = FLAVORS

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        backend: Backend = "numpy",
    ):
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if backend not in FLAVORS:
            raise ValueError(f"Invalid backend: {backend}")
        self.rotation_type = rotation_type
        self.backend = backend
        self._backend = get_backend(backend)

        resolved_path = get_model_path(model_path)
        self.weights = load_model_data(resolved_path, simplify=simplify)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(FLAME_JOINT_NAMES)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V 5"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[np.ndarray, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def exprdirs(self) -> Float[np.ndarray, "V 3 E"]:
        return self.weights.exprdirs

    @property
    def posedirs(self) -> Float[np.ndarray, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[np.ndarray, "V 5"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 S"],
        expression: Float[np.ndarray, "B E"] | None = None,
        pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"] | None = None,
        head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        batch_size = shape.shape[0] if shape.ndim > 1 and shape.shape[0] > 1 else 1
        if pose is not None:
            batch_size = pose.shape[0]
        if expression is None:
            expression = np.zeros((batch_size, 100), dtype=np.float32)
        if pose is None:
            pose = SO3.identity_as(
                expression,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            )
        return self._backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 S"],
        expression: Float[np.ndarray, "B E"] | None = None,
        pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"] | None = None,
        head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 5 4 4"]:
        batch_size = shape.shape[0] if shape.ndim > 1 and shape.shape[0] > 1 else 1
        if pose is not None:
            batch_size = pose.shape[0]
        if expression is None:
            expression = np.zeros((batch_size, 100), dtype=np.float32)
        if pose is None:
            pose = SO3.identity_as(
                expression,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            )
        return self._backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        ref = np.zeros((batch_size, 100), dtype=dtype)
        return {
            "shape": np.zeros((1, 300), dtype=dtype),
            "expression": np.zeros((batch_size, 100), dtype=dtype),
            "pose": SO3.identity_as(
                ref,
                batch_dims=(batch_size, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "head_rotation": SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type=self.rotation_type, xp=np),
            "global_rotation": SO3.identity_as(ref, batch_dims=(batch_size,), rotation_type=self.rotation_type, xp=np),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }


def get_backend(backend: Backend):
    if backend == "numpy":
        return numpy_backend
    if backend == "scipy":
        return scipy_backend

    try:
        from body_models.flame.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use FLAME backend='numba'.") from exc

    return numba_backend
