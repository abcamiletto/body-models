"""NumPy backend for SMPL-H model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from body_models.base import BodyModel
from nanomanifold import SO3

from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from body_models.smplh.backends import numpy as numpy_backend
from body_models.smplh.backends import scipy as scipy_backend
from body_models.smplh.io import get_model_path, load_model_data
from body_models.smplh.constants import SMPLH_BODY_PRESETS, SMPLH_HAND_PRESETS, SMPLH_JOINTS

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
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "scipy", "numba"] = "numpy",
    ):
        """Initialize the SMPLH model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            gender: Model gender variant to load.
            flat_hand_mean: Whether to use a flat hand as the pose mean.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
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
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
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
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
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
        batch_dims: tuple[int, ...] = (),
        dtype=np.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        body_pose_ref = np.zeros((*batch_dims, self.NUM_BODY_JOINTS, 3), dtype=dtype)
        hand_pose_ref = np.zeros((*batch_dims, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        pelvis_ref = np.zeros((*batch_dims, 3), dtype=dtype)
        params = {
            "shape": np.zeros((*batch_dims, 10), dtype=dtype),
            "body_pose": SO3.identity_as(
                body_pose_ref,
                batch_dims=(*batch_dims, self.NUM_BODY_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(*batch_dims, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "pelvis_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_rotation": SO3.identity_as(
                pelvis_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }
        if hands != "default":
            params["hand_pose"] = self._hand_preset(batch_dims, dtype, hands)
        return params

    def _hand_preset(self, batch_dims: tuple[int, ...], dtype, hands: str):
        preset = SMPLH_HAND_PRESETS[hands]
        axis_angle = np.asarray(preset, dtype=dtype).reshape(self.NUM_HAND_JOINTS, 3)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        return SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = np.asarray(SMPLH_BODY_PRESETS["a_pose"], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
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
