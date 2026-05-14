"""NumPy frontend for ANNY."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.anny import pose as pose_utils
from body_models.anny.backends import numpy as numpy_backend
from body_models.anny.io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy
from body_models.anny.constants import ANNY_IPOSE, ANNY_JOINTS, ANNY_TPOSE
from body_models.base import BodyModel
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["ANNY"]


class ANNY(BodyModel):
    """ANNY body model with NumPy backend."""

    has_hands = True

    kernels = ("numpy", "numba")
    JOINTS = ANNY_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
        rotation_type: RotationType = "axis_angle",
        kernel: Literal["numpy", "numba"] = "numpy",
    ) -> None:
        if rig not in ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo"):
            raise ValueError(f"Invalid rig: {rig}")
        if topology not in ("default", "makehuman"):
            raise ValueError(f"Invalid topology: {topology}")
        if simplify < 1.0:
            raise ValueError("simplify must be >= 1.0")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        if kernel not in self.kernels:
            raise ValueError(f"Invalid kernel: {kernel}")

        self.weights = load_model_data_numpy(model_path, rig=rig, topology=topology, simplify=simplify)
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

    @property
    def faces(self) -> Int[np.ndarray, "F _"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.bone_labels)

    @property
    def joint_names(self) -> list[str]:
        return list(self.weights.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.weights.template_vertices.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.weights.template_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        gender: Float[np.ndarray, "B"],
        age: Float[np.ndarray, "B"],
        muscle: Float[np.ndarray, "B"],
        weight: Float[np.ndarray, "B"],
        height: Float[np.ndarray, "B"],
        proportions: Float[np.ndarray, "B"],
        body_pose: Float[np.ndarray, "B 64 N"] | Float[np.ndarray, "B 64 3 3"],
        head_pose: Float[np.ndarray, "B 60 N"] | Float[np.ndarray, "B 60 3 3"],
        hand_pose: Float[np.ndarray, "B 38 N"] | Float[np.ndarray, "B 38 3 3"],
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[np.ndarray, "B V 3"]:
        pose = pose_utils.pack_pose(np, global_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.forward_vertices(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def forward_skeleton(
        self,
        gender: Float[np.ndarray, "B"],
        age: Float[np.ndarray, "B"],
        muscle: Float[np.ndarray, "B"],
        weight: Float[np.ndarray, "B"],
        height: Float[np.ndarray, "B"],
        proportions: Float[np.ndarray, "B"],
        body_pose: Float[np.ndarray, "B 64 N"] | Float[np.ndarray, "B 64 3 3"],
        head_pose: Float[np.ndarray, "B 60 N"] | Float[np.ndarray, "B 60 3 3"],
        hand_pose: Float[np.ndarray, "B 38 N"] | Float[np.ndarray, "B 38 3 3"],
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        pose = pose_utils.pack_pose(np, global_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.forward_skeleton(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=np.float32,
        hands: Literal["open", "rest"] = "rest",
    ) -> dict[str, np.ndarray]:
        if hands not in ("open", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'open' or 'rest'.")

        pose_ref = np.zeros((batch_size,), dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, self.num_joints),
            rotation_type=self.rotation_type,
            xp=np,
        )
        global_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(np, pose)
        return {
            **{
                name: np.full((batch_size,), 0.5, dtype=dtype)
                for name in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        pose_parts = (
            params["global_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        pose = pose_utils.pack_pose(np, *pose_parts)
        for joint_name, values in ANNY_TPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=np)
            pose = common.set(pose, (slice(None), index), converted, xp=np)
        global_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(np, pose)
        params.update(
            body_pose=body_pose,
            head_pose=head_pose,
            hand_pose=hand_pose,
            global_rotation=global_rotation,
        )
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)

    def get_ipose(
        self,
        batch_size: int = 1,
        hands: Literal["open", "rest"] = "rest",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        params = self.get_rest_pose(batch_size=batch_size, hands=hands, **kwargs)
        pose_parts = (
            params["global_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        pose = pose_utils.pack_pose(np, *pose_parts)
        for joint_name, values in ANNY_IPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=np)
            pose = common.set(pose, (slice(None), index), converted, xp=np)
        global_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(np, pose)
        params.update(
            body_pose=body_pose,
            head_pose=head_pose,
            hand_pose=hand_pose,
            global_rotation=global_rotation,
        )
        return params


def _get_kernel(kernel: Literal["numpy", "numba"]):
    if kernel == "numpy":
        return numpy_backend

    try:
        from body_models.anny.backends import numba as numba_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[numba] to use ANNY kernel='numba'.") from exc

    return numba_backend
