"""NumPy backend for MANO model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.mano.backends import numpy as numpy_backend
from body_models.mano.backends import scipy as scipy_backend
from body_models.mano.io import get_model_path, load_model_data
from body_models.mano.constants import LEFT_MANO_JOINTS, RIGHT_MANO_JOINTS
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["MANO"]


class MANO(BodyModel):
    """MANO hand model with NumPy backend."""

    NUM_HAND_JOINTS = 15
    NUM_JOINTS = 16
    kernels = ("numpy", "scipy", "numba")

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "scipy", "numba"] = "numpy",
    ):
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side}. Must be 'right' or 'left'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")

        self.side = side if side is not None else "right"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

        resolved_path = get_model_path(model_path, side)
        self.weights = load_model_data(resolved_path, flat_hand_mean=flat_hand_mean, simplify=simplify)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def common_joints(self):
        return LEFT_MANO_JOINTS if self.side == "left" else RIGHT_MANO_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V 16"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.v_template

    @property
    def shapedirs(self) -> Float[np.ndarray, "V 3 S"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[np.ndarray, "P V*3"]:
        return self.weights.posedirs

    @property
    def lbs_weights(self) -> Float[np.ndarray, "V 16"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        hand_pose: Float[np.ndarray, "B 15 N"] | Float[np.ndarray, "B 15 3 3"],
        wrist_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        hand_pose: Float[np.ndarray, "B 15 N"] | Float[np.ndarray, "B 15 3 3"],
        wrist_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 16 4 4"]:
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=np.float32,
        hands: Literal["rest"] = "rest",
    ) -> dict[str, np.ndarray]:
        if hands != "rest":
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'rest'.")

        hand_pose_ref = np.zeros((batch_size, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        wrist_ref = np.zeros((batch_size, 3), dtype=dtype)
        return {
            "shape": np.zeros((1, 10), dtype=dtype),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(batch_size, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "wrist_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }


def _get_kernel(kernel: Literal["numpy", "scipy", "numba"]):
    if kernel == "numpy":
        return numpy_backend
    if kernel == "scipy":
        return scipy_backend

    try:
        from body_models.mano.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use MANO kernel='numba'.") from exc

    return numba_backend
