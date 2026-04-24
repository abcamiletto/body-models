"""NumPy backend for the GarmentMeasurements body model."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import load_model_data

__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")

        data = load_model_data(model_path=model_path, dtype=np.float32)
        self.mean_vertices = data["mean_vertices"]
        self.components = data["components"]
        self.eigenvalues = data["eigenvalues"]
        self.bind_quats = data["bind_quats"]
        self._skin_weights = data["skin_weights"]
        self.mvc_weights = data["mvc_weights"]
        self._faces = data["faces"]
        self.parents = data["parents"].astype(int).tolist()
        self._joint_names = list(data["joint_names"])
        self.rotation_type = rotation_type

    @property
    def faces(self) -> Int[np.ndarray, "F _"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_vertices.shape[0]

    @property
    def num_shape_components(self) -> int:
        return self.eigenvalues.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self._skin_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.mean_vertices

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B C"],
        pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        return core.forward_vertices(
            mean_vertices=self.mean_vertices,
            components=self.components,
            eigenvalues=self.eigenvalues,
            bind_quats=self.bind_quats,
            skin_weights=self._skin_weights,
            mvc_weights=self.mvc_weights,
            parents=self.parents,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            xp=np,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B C"],
        pose: Float[np.ndarray, "B J N"] | Float[np.ndarray, "B J 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        return core.forward_skeleton(
            mean_vertices=self.mean_vertices,
            components=self.components,
            eigenvalues=self.eigenvalues,
            bind_quats=self.bind_quats,
            mvc_weights=self.mvc_weights,
            parents=self.parents,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            xp=np,
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
