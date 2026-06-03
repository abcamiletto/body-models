"""NumPy backend for the GarmentMeasurements PCA body model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import BodyModel
from body_models.rotations import VALID_ROTATION_TYPES, RotationType
from .backends import numpy as numpy_backend
from .backends.core import GarmentMeasurementsIdentity, GarmentMeasurementsPreparedPose
from .io import get_model_path, load_model_data
from .constants import GARMENT_BODY_PRESETS, GARMENT_HAND_PRESETS, GARMENT_JOINTS
from .pose import pack_pose, unpack_pose


__all__ = ["GarmentMeasurements"]


class GarmentMeasurements(BodyModel):
    """GarmentMeasurements PCA body model with FBX-derived skeleton/skinning."""

    has_hands = True
    has_head = True
    kernels = ("numpy", "numba")
    JOINTS = GARMENT_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "numba"] = "numpy",
    ) -> None:
        """Initialize the GarmentMeasurements model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")

        self.weights = load_model_data(get_model_path(model_path), dtype=np.float32)
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)

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
        return [int(parent) for parent in self.weights.parents.tolist()]

    def forward_vertices(
        self,
        body_pose: Float[np.ndarray, "*batch 25 N"] | Float[np.ndarray, "*batch 25 3 3"],
        head_pose: Float[np.ndarray, "*batch 3 N"] | Float[np.ndarray, "*batch 3 3 3"],
        hand_pose: Float[np.ndarray, "*batch 30 N"] | Float[np.ndarray, "*batch 30 3 3"],
        pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"],
        global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
        global_translation: Float[np.ndarray, "*batch 3"] | None = None,
        vertex_indices: list[int] | None = None,
        *,
        shape: Float[np.ndarray, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
    ) -> Float[np.ndarray, "*batch V 3"]:
        """Compute posed mesh vertices.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.

        Returns:
            Posed vertex positions.
        """
        pose = pack_pose(np, pelvis_rotation, body_pose, head_pose, hand_pose)
        if identity is None:
            assert shape is not None
            batch_shape = pose.shape[: -(self.num_rot_dims + 1)]
            shape = np.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)
        pose = self.prepare_pose(pose, identity=identity)
        return self._kernel.forward_vertices(
            weights=self.weights,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=pose["skinning_transforms"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[np.ndarray, "*batch 25 N"] | Float[np.ndarray, "*batch 25 3 3"],
        head_pose: Float[np.ndarray, "*batch 3 N"] | Float[np.ndarray, "*batch 3 3 3"],
        hand_pose: Float[np.ndarray, "*batch 30 N"] | Float[np.ndarray, "*batch 30 3 3"],
        pelvis_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"],
        global_rotation: Float[np.ndarray, "*batch N"] | Float[np.ndarray, "*batch 3 3"] | None = None,
        global_translation: Float[np.ndarray, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
        *,
        shape: Float[np.ndarray, "*batch C"] | None = None,
        identity: GarmentMeasurementsIdentity | None = None,
    ) -> Float[np.ndarray, "*batch J 4 4"]:
        """Compute posed joint transforms.

        Args:
            shape: Shape coefficients.
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            pelvis_rotation: Root pelvis rotation.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.

        Returns:
            Joint transforms in the model hierarchy.
        """
        pose = pack_pose(np, pelvis_rotation, body_pose, head_pose, hand_pose)
        if identity is None:
            assert shape is not None
            batch_shape = pose.shape[: -(self.num_rot_dims + 1)]
            shape = np.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        pose = self.prepare_pose(pose, identity=identity, skip_vertices=True)
        return self._kernel.forward_skeleton(
            self.weights,
            pose["skeleton_transforms"],
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def prepare_identity(
        self,
        shape: Float[np.ndarray, "*batch C"],
        skip_vertices: bool = False,
    ) -> GarmentMeasurementsIdentity:
        """Precompute shape-dependent state for repeated forward passes."""
        return self._kernel.prepare_identity(self.weights, shape, skip_vertices=skip_vertices)

    def prepare_pose(
        self,
        pose: Float[np.ndarray, "*batch J N"] | Float[np.ndarray, "*batch J 3 3"],
        *,
        identity: GarmentMeasurementsIdentity,
        skip_vertices: bool = False,
    ) -> GarmentMeasurementsPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        return self._kernel.prepare_pose(
            self.weights,
            pose,
            rotation_type=self.rotation_type,
            bind_skeleton=identity["bind_skeleton"],
            local_bind_translations=identity["local_bind_translations"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype=np.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        pose_ref = np.zeros((*batch_dims, self.num_joints, 3), dtype=dtype)
        global_ref = np.zeros((*batch_dims,), dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=np,
        )
        pelvis_rotation, body_pose, head_pose, hand_pose = unpack_pose(np, pose)
        if hands != "default":
            axis_angle = np.asarray(GARMENT_HAND_PRESETS[hands], dtype=dtype).reshape(-1, 3)
            axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
        return {
            "shape": np.zeros((*batch_dims, self.num_shape_components), dtype=dtype),
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = np.asarray(GARMENT_BODY_PRESETS["t_pose"], dtype=params["body_pose"].dtype)
        axis_angle = np.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=np).copy()
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


def _get_kernel(kernel: Literal["numpy", "numba"]):
    if kernel == "numpy":
        return numpy_backend

    try:
        from body_models.bodies.garment_measurements.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use GarmentMeasurements kernel='numba'.") from exc

    return numba_backend
