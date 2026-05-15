"""NumPy backend for MHR model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from body_models.mhr.backends import numpy as backend
from body_models.mhr.constants import (
    MHR_BODY_POSE_DIM,
    MHR_HAND_PRESETS,
    MHR_HAND_POSE_DIM,
    MHR_BODY_PRESETS,
    MHR_JOINTS,
)
from body_models.mhr.io import get_model_path, load_model_data
from body_models.mhr.pose import pack_pose

__all__ = ["MHR"]


class MHR(BodyModel):
    """MHR body model with NumPy backend."""

    has_hands = True

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
    def body_pose_dim(self) -> int:
        return MHR_BODY_POSE_DIM

    @property
    def hand_pose_dim(self) -> int:
        return MHR_HAND_POSE_DIM

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
        body_pose: Float[np.ndarray, "B 100"],
        hand_pose: Float[np.ndarray, "B 104"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=pack_pose(np, body_pose, hand_pose),
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 45"],
        body_pose: Float[np.ndarray, "B 100"],
        hand_pose: Float[np.ndarray, "B 104"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=pack_pose(np, body_pose, hand_pose),
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=np.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        hand_pose = np.zeros((*batch_dims, self.hand_pose_dim), dtype=dtype)
        if hands != "default":
            hand_pose = np.asarray(MHR_HAND_PRESETS[hands], dtype=dtype).reshape(self.hand_pose_dim)
            hand_pose = np.broadcast_to(hand_pose, (*batch_dims, self.hand_pose_dim)).copy()
        return {
            "shape": np.zeros((*batch_dims, self.SHAPE_DIM), dtype=dtype),
            "body_pose": np.zeros((*batch_dims, self.body_pose_dim), dtype=dtype),
            "hand_pose": hand_pose,
            "expression": np.zeros((*batch_dims, self.EXPR_DIM), dtype=dtype),
            "global_rotation": np.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        body_pose = np.asarray(MHR_BODY_PRESETS["t_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = np.broadcast_to(body_pose, (*batch_dims, *body_pose.shape)).copy()
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
