"""PyTorch frontend for ANNY."""

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.anny import pose as pose_utils
from body_models.anny.backends import torch as torch_backend
from body_models.anny.backends.core import AnnyIdentity, AnnyPreparedPose
from body_models.anny.io import EXCLUDED_PHENOTYPES, PHENOTYPE_LABELS, load_model_data_numpy
from body_models.anny.constants import ANNY_BODY_PRESETS, ANNY_HAND_PRESETS, ANNY_JOINTS
from body_models.base import BodyModel
from body_models.rotations import VALID_ROTATION_TYPES, RotationType

__all__ = ["ANNY"]


class ANNY(BodyModel, nn.Module):
    """ANNY body model with PyTorch backend."""

    has_hands = True
    has_head = True
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
        """Initialize the ANNY model.

        Args:
            model_path: Path to model assets, or the default assets when omitted.
            rig: Rig variant to load.
            topology: Mesh topology variant to load.
            all_phenotypes: Whether to expose the full phenotype control set.
            extrapolate_phenotypes: Whether phenotype values may extend beyond the trained range.
            simplify: Mesh simplification factor to apply while loading.
            rotation_type: Rotation representation expected by pose inputs.
            kernel: Backend kernel used for forward evaluation.
        """
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
        body_pose: Float[Tensor, "B 64 N"] | Float[Tensor, "B 64 3 3"],
        head_pose: Float[Tensor, "B 60 N"] | Float[Tensor, "B 60 3 3"],
        hand_pose: Float[Tensor, "B 38 N"] | Float[Tensor, "B 38 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 6"] | None = None,
        identity: AnnyIdentity | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute posed mesh vertices.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            vertex_indices: Optional subset of vertices to return.
            shape: Packed phenotype controls.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Posed vertex positions.
        """
        if identity is None:
            assert shape is not None
            pose = pose_utils.pack_pose(torch, global_rotation, body_pose, head_pose, hand_pose)
            batch_shape = tuple(pose.shape[: -(self.num_rot_dims + 1)])
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape)
        prepared_pose = self.prepare_pose(body_pose, head_pose, hand_pose, global_rotation, identity=identity)
        return self._kernel.forward_vertices(
            self.weights,
            identity["rest_vertices"],
            prepared_pose["skinning_transforms"],
            global_translation=global_translation,
            vertex_indices=vertex_indices,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Tensor, "B 64 N"] | Float[Tensor, "B 64 3 3"],
        head_pose: Float[Tensor, "B 60 N"] | Float[Tensor, "B 60 3 3"],
        hand_pose: Float[Tensor, "B 38 N"] | Float[Tensor, "B 38 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices: Any | None = None,
        *,
        shape: Float[Tensor, "*batch 6"] | None = None,
        identity: AnnyIdentity | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute posed joint transforms.

        Args:
            body_pose: Local body joint rotations.
            head_pose: Local head and facial joint rotations.
            hand_pose: Local hand joint rotations.
            global_rotation: Global model rotation.
            global_translation: Global model translation.
            joint_indices: Optional subset of joints to return.
            shape: Packed phenotype controls.
            identity: Optional output from :meth:`prepare_identity`.

        Returns:
            Joint transforms in the model hierarchy.
        """
        if identity is None:
            assert shape is not None
            pose = pose_utils.pack_pose(torch, global_rotation, body_pose, head_pose, hand_pose)
            batch_shape = tuple(pose.shape[: -(self.num_rot_dims + 1)])
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))
            identity = self.prepare_identity(shape, skip_vertices=True)
        prepared_pose = self.prepare_pose(
            body_pose,
            head_pose,
            hand_pose,
            global_rotation,
            identity=identity,
            skip_vertices=True,
        )
        return self._kernel.forward_skeleton(
            self.weights,
            prepared_pose["skeleton_transforms"],
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def prepare_identity(
        self,
        shape: Float[Tensor, "*batch 6"],
        skip_vertices: bool = False,
    ) -> AnnyIdentity:
        """Precompute phenotype-dependent state for repeated forward passes."""
        return self._kernel.prepare_identity(
            self.weights,
            shape,
            extrapolate_phenotypes=self.extrapolate_phenotypes,
            skip_vertices=skip_vertices,
        )

    def phenotype_to_shape(
        self,
        gender: Float[Tensor, "*batch"],
        age: Float[Tensor, "*batch"],
        muscle: Float[Tensor, "*batch"],
        weight: Float[Tensor, "*batch"],
        height: Float[Tensor, "*batch"],
        proportions: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*batch 6"]:
        """Pack named phenotype controls into the ANNY shape vector."""
        return torch.stack([gender, age, muscle, weight, height, proportions], dim=-1)

    def prepare_pose(
        self,
        body_pose: Float[Tensor, "B 64 N"] | Float[Tensor, "B 64 3 3"],
        head_pose: Float[Tensor, "B 60 N"] | Float[Tensor, "B 60 3 3"],
        hand_pose: Float[Tensor, "B 38 N"] | Float[Tensor, "B 38 3 3"],
        global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"],
        *,
        identity: AnnyIdentity,
        skip_vertices: bool = False,
    ) -> AnnyPreparedPose:
        """Precompute pose-dependent state for repeated forward passes."""
        pose = pose_utils.pack_pose(torch, global_rotation, body_pose, head_pose, hand_pose)
        return self._kernel.prepare_pose(
            self.weights,
            pose,
            rotation_type=self.rotation_type,
            rest_skeleton_transforms=identity["rest_skeleton_transforms"],
            skip_vertices=skip_vertices,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: torch.dtype | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Tensor]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        dtype = dtype or self.weights.template_vertices.dtype
        device = self.weights.template_vertices.device
        pose_ref = torch.zeros((*batch_dims,), device=device, dtype=dtype)
        pose = SO3.identity_as(
            pose_ref,
            batch_dims=(*batch_dims, self.num_joints),
            rotation_type=self.rotation_type,
            xp=torch,
        )
        global_rotation, body_pose, head_pose, hand_pose = pose_utils.unpack_pose(torch, pose)
        if hands != "default":
            preset = ANNY_HAND_PRESETS[hands]
            axis_angle = torch.asarray(preset, device=device, dtype=dtype).reshape(-1, 3)
            axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
            hand_pose = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return {
            "shape": torch.full((*batch_dims, 6), 0.5, device=device, dtype=dtype),
            "body_pose": body_pose,
            "head_pose": head_pose,
            "hand_pose": hand_pose,
            "global_rotation": global_rotation,
            "global_translation": torch.zeros((*batch_dims, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)
        axis_angle = torch.as_tensor(
            ANNY_BODY_PRESETS["t_pose"], device=params["body_pose"].device, dtype=params["body_pose"].dtype
        )
        axis_angle = torch.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=torch)
        return params

    def get_apose(
        self,
        batch_dims: tuple[int, ...] = (),
        hands: Literal["default", "flat", "rest"] = "default",
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_dims=batch_dims, hands=hands, **kwargs)


def _get_kernel(kernel: Literal["torch", "warp"]):
    if kernel == "torch":
        return torch_backend

    try:
        from body_models.anny.backends import warp as warp_backend
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install body-models[warp] to use ANNY kernel='warp'.") from exc

    return warp_backend
