"""NumPy backend for SMPL-H model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models import common
from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplh.backends import numpy as numpy_backend
from body_models.smplh.backends import scipy as scipy_backend
from body_models.smplh.io import get_model_path, load_model_data
from body_models.smplh.constants import SMPLH_APOSE, SMPLH_IPOSE, SMPLH_JOINTS

__all__ = ["SMPLH"]


class SMPLH(BodyModel):
    """SMPL-H body model with NumPy backend."""

    has_hands = True

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_JOINTS = 52
    kernels = ("numpy", "scipy", "numba")
    JOINTS = SMPLH_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: Literal["neutral", "male", "female"] | None = None,
        flat_hand_mean: bool = True,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "scipy", "numba"] = "numpy",
    ):
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")

        self.gender = gender if gender is not None else "neutral"
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

        resolved_path = get_model_path(model_path, gender)
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
    def num_vertices(self) -> int:
        return self.weights.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V 52"]:
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
    def lbs_weights(self) -> Float[np.ndarray, "V 52"]:
        return self.weights.lbs_weights

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        body_pose: Float[np.ndarray, "B 21 N"] | Float[np.ndarray, "B 21 3 3"],
        hand_pose: Float[np.ndarray, "B 30 N"] | Float[np.ndarray, "B 30 3 3"],
        pelvis_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        return self._kernel.forward_vertices(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 10"],
        body_pose: Float[np.ndarray, "B 21 N"] | Float[np.ndarray, "B 21 3 3"],
        hand_pose: Float[np.ndarray, "B 30 N"] | Float[np.ndarray, "B 30 3 3"],
        pelvis_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B 52 4 4"]:
        return self._kernel.forward_skeleton(
            weights=self.weights,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=np.float32,
        hands: Literal["open", "rest"] = "rest",
    ) -> dict[str, np.ndarray]:
        if hands not in ("open", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'open' or 'rest'.")

        body_pose_ref = np.zeros((batch_size, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        hand_pose_ref = np.zeros((batch_size, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        pelvis_ref = np.zeros((batch_size, 3), dtype=dtype)
        params = {
            "shape": np.zeros((1, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(batch_size, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(batch_size, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
        if hands == "open":
            params["hand_pose"] = self._open_hand_pose(params["hand_pose"])
        return params

    def _open_hand_pose(self, hand_pose: Float[np.ndarray, "B 30 N"] | Float[np.ndarray, "B 30 3 3"]):
        hand_mean = np.asarray(self.weights.hand_mean.reshape(-1, 3), dtype=hand_pose.dtype)
        template = hand_pose[:, :, 0, :] if hand_pose.ndim == 4 else hand_pose
        axis_angle = np.zeros_like(template) - hand_mean
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np)

    def get_tpose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPLH_APOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=np)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=np)
        params["body_pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        body_pose = params["body_pose"]
        for index, values in SMPLH_IPOSE.items():
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=np)
            body_pose = common.set(body_pose, (slice(None), index), converted, xp=np)
        params["body_pose"] = body_pose
        return params


def _get_kernel(kernel: Literal["numpy", "scipy", "numba"]):
    if kernel == "numpy":
        return numpy_backend
    if kernel == "scipy":
        return scipy_backend

    try:
        from body_models.smplh.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use SMPLH kernel='numba'.") from exc

    return numba_backend
