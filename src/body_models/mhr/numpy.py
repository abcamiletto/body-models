"""NumPy backend for MHR model."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from body_models.mhr.backends import numpy as backend
from body_models.mhr.io import get_model_path, load_model_data
from body_models.mhr.constants import MHR_IPOSE_TARGETS, MHR_JOINTS, MHR_TPOSE_TARGETS

__all__ = ["MHR"]


class MHR(BodyModel):
    """MHR body model with NumPy backend."""

    SHAPE_DIM = 45
    EXPR_DIM = 72
    JOINTS = MHR_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        self.weights = load_model_data(get_model_path(model_path), lod=lod, simplify=simplify)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.parents)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.weights.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        dense = np.zeros((self.weights.skin_weights.shape[0], self.num_joints), dtype=self.weights.skin_weights.dtype)
        dense[np.arange(self.weights.skin_weights.shape[0])[:, None], self.weights.skin_indices] = (
            self.weights.skin_weights
        )
        return dense

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 45"],
        pose: Float[np.ndarray, "B 204"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 45"],
        pose: Float[np.ndarray, "B 204"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        return {
            "shape": np.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": np.zeros((batch_size, self.pose_dim), dtype=dtype),
            "expression": np.zeros((batch_size, self.EXPR_DIM), dtype=dtype),
            "global_rotation": np.zeros((batch_size, 3), dtype=dtype),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        targets = MHR_TPOSE_TARGETS
        pose = params["pose"]
        rows = [
            next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name) * 7 + component
            for joint_name, component, _ in targets
        ]
        values = np.asarray([value for _, _, value in targets], dtype=pose.dtype)
        transform = np.asarray(self.weights.parameter_transform, dtype=pose.dtype)
        system = transform[rows, : self.pose_dim]
        params["pose"] = common.set(pose, (slice(None),), np.linalg.pinv(system) @ values, xp=np)
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_size=batch_size, **kwargs)

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        targets = MHR_IPOSE_TARGETS
        pose = params["pose"]
        rows = [
            next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name) * 7 + component
            for joint_name, component, _ in targets
        ]
        values = np.asarray([value for _, _, value in targets], dtype=pose.dtype)
        transform = np.asarray(self.weights.parameter_transform, dtype=pose.dtype)
        system = transform[rows, : self.pose_dim]
        params["pose"] = common.set(pose, (slice(None),), np.linalg.pinv(system) @ values, xp=np)
        return params
