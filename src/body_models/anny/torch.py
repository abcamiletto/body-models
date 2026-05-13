"""PyTorch frontend for ANNY."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.anny import pose as pose_utils
from body_models.anny.backends import torch as torch_backend
from body_models.anny.io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy
from body_models.anny.constants import ANNY_IPOSE, ANNY_JOINTS, ANNY_TPOSE
from body_models.base import BodyModel
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["ANNY"]


class ANNY(BodyModel, nn.Module):
    """ANNY body model with PyTorch backend."""

    kernels = ("torch", "warp")
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
        kernel: Literal["torch", "warp"] = "torch",
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
        super().__init__()

        data = load_model_data_numpy(model_path, rig=rig, topology=topology, simplify=simplify)
        self.weights = common.torchify(data)
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self._kernel = _get_kernel(kernel)
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

    @property
    def faces(self) -> Int[Tensor, "F _"]:
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
    def dtype(self) -> torch.dtype:
        return self.weights.template_vertices.dtype

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self.weights.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.template_vertices

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    def forward_vertices(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        body_pose: Float[Tensor, "B 64 N"] | Float[Tensor, "B 64 3 3"],
        head_pose: Float[Tensor, "B 60 N"] | Float[Tensor, "B 60 3 3"],
        hand_pose: Float[Tensor, "B 38 N"] | Float[Tensor, "B 38 3 3"],
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
        pose = pose_utils.pack_pose(torch, pelvis_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.forward_vertices(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def forward_skeleton(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        body_pose: Float[Tensor, "B 64 N"] | Float[Tensor, "B 64 3 3"],
        head_pose: Float[Tensor, "B 60 N"] | Float[Tensor, "B 60 3 3"],
        hand_pose: Float[Tensor, "B 38 N"] | Float[Tensor, "B 38 3 3"],
        pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B J 4 4"]:
        pose = pose_utils.pack_pose(torch, pelvis_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.forward_skeleton(
            weights=self.weights,
            gender=gender,
            age=age,
            muscle=muscle,
            weight=weight,
            height=height,
            proportions=proportions,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
        dtype = dtype or self.weights.template_vertices.dtype
        device = self.weights.template_vertices.device
        pose_ref = torch.zeros((batch_size,), device=device, dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, self.num_joints),
            rotation_type=self.rotation_type,
            xp=torch,
        )
        pelvis_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(torch, pose)
        return {
            **{
                name: torch.full((batch_size,), 0.5, device=device, dtype=dtype)
                for name in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=torch,
            ),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        pose = pose_utils.pack_pose(
            torch, params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"]
        )
        for joint_name, values in ANNY_TPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=torch)
            converted = torch.as_tensor(converted, device=pose.device, dtype=pose.dtype)
            pose = common.set(pose, (slice(None), index), converted, xp=torch)
        params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"] = (
            pose_utils.unpack_pose(torch, pose)
        )
        return params

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_size=batch_size, **kwargs)

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        pose = pose_utils.pack_pose(
            torch, params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"]
        )
        for joint_name, values in ANNY_IPOSE.items():
            index = next(i for i, name in enumerate(self.joint_names) if name.lower() == joint_name)
            converted = SO3.convert(values, src="axis_angle", dst=self.rotation_type, xp=torch)
            converted = torch.as_tensor(converted, device=pose.device, dtype=pose.dtype)
            pose = common.set(pose, (slice(None), index), converted, xp=torch)
        params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"] = (
            pose_utils.unpack_pose(torch, pose)
        )
        return params


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.anny.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use ANNY kernel='warp'.") from exc

    return warp_backend
