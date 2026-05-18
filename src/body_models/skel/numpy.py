"""NumPy backend for SKEL model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from body_models.skel.backends import numpy as backend
from body_models.skel.io import get_model_path, load_model_data
from body_models.skel.constants import SKEL_BODY_PRESETS, SKEL_JOINTS

__all__ = ["SKEL"]


class SKEL(BodyModel):
    """SKEL body model with NumPy backend."""

    NUM_BETAS = 10
    NUM_JOINTS = 24
    NUM_POSE_PARAMS = 46
    JOINTS = SKEL_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["male", "female"] | None = None,
        simplify: float = 1.0,
    ):
        if gender not in {"male", "female"}:
            raise ValueError(f"Invalid gender: {gender}. Must be 'male' or 'female'.")
        assert simplify >= 1.0

        self.gender = gender
        self.weights = load_model_data(get_model_path(model_path, gender), simplify=simplify)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.joint_names)

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.v_template + self.weights.feet_offset

    @property
    def shapedirs(self) -> Float[np.ndarray, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[np.ndarray, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def _feet_offset(self) -> Float[np.ndarray, "3"]:
        return self.weights.feet_offset

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        body_pose: Float[np.ndarray, "B 46"],
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        """Evaluate posed mesh vertices."""
        return backend.forward_vertices(
            weights=self.weights,
            shape=shape,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        body_pose: Float[np.ndarray, "B 46"],
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 24 4 4"]:
        """Evaluate world-space joint transforms."""
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=np.float32) -> dict[str, np.ndarray]:
        """Return default parameters for this model."""
        return {
            "shape": np.zeros((*batch_dims, self.NUM_BETAS), dtype=dtype),
            "body_pose": np.zeros((*batch_dims, self.NUM_POSE_PARAMS), dtype=dtype),
            "global_rotation": np.zeros((*batch_dims, 3), dtype=dtype),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Return parameters for the canonical T-pose."""
        return self.get_rest_pose(batch_dims=batch_dims, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Return parameters for the canonical A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        body_pose = np.asarray(SKEL_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        params["body_pose"] = np.broadcast_to(body_pose, (*batch_dims, *body_pose.shape)).copy()
        return params
