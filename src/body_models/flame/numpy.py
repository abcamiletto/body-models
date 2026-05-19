"""NumPy backend for FLAME model."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.flame.backends import numpy as numpy_backend
from body_models.flame.backends import scipy as scipy_backend
from body_models.flame.constants import FLAME_JOINT_NAMES
from body_models.flame.io import get_model_path, load_model_data
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["FLAME"]


class FLAME(BodyModel):
    """FLAME head model with NumPy backend."""

    NUM_HEAD_JOINTS = 4
    NUM_JOINTS = 5
    kernels = ("numpy", "scipy", "numba")

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "scipy", "numba"] = "numpy",
    ):
        """Initialize the FLAME model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

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
        expression: Float[np.ndarray, "B E"],
        head_pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
        head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            expression: Facial expression coefficients.
            head_pose: Local head and facial joint rotations.
            head_rotation: Root head rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 S"],
        expression: Float[np.ndarray, "B E"],
        head_pose: Float[np.ndarray, "B 4 N"] | Float[np.ndarray, "B 4 3 3"],
        head_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: Any | None = None,
    ) -> Float[np.ndarray, "B 5 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            expression: Facial expression coefficients.
            head_pose: Local head and facial joint rotations.
            head_rotation: Root head rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            expression=expression,
            pose=head_pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        ref = np.zeros((*batch_dims, 100), dtype=dtype)
        return {
            "shape": np.zeros((*batch_dims, 300), dtype=dtype),
            "expression": np.zeros((*batch_dims, 100), dtype=dtype),
            "head_pose": SO3.identity_as(
                ref,
                batch_dims=(*batch_dims, self.NUM_HEAD_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "head_rotation": SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=self.rotation_type, xp=np),
            "global_rotation": SO3.identity_as(ref, batch_dims=batch_dims, rotation_type=self.rotation_type, xp=np),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }


def _get_kernel(kernel: Literal["numpy", "scipy", "numba"]):
    if kernel == "numpy":
        return numpy_backend
    if kernel == "scipy":
        return scipy_backend

    try:
        from body_models.flame.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use FLAME kernel='numba'.") from exc

    return numba_backend
