"""PyTorch backend for SKEL model."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.base import BodyModel
from body_models.skel.backends import torch as backend
from body_models.skel.backends import core
from body_models.skel.io import get_model_path, load_model_data
from body_models.skel.constants import SKEL_APOSE, SKEL_IPOSE, SKEL_JOINTS

__all__ = ["SKEL"]


class SKEL(BodyModel, nn.Module):
    """SKEL body model with PyTorch backend."""

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
        super().__init__()

        self.gender = gender
        data = load_model_data(get_model_path(model_path, gender), simplify=simplify)
        self.weights = common.torchify(data)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
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
    def skin_weights(self) -> Float[Tensor, "V 24"]:
        return self.weights.skin_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.weights.v_template + self.weights.feet_offset

    @property
    def shapedirs(self) -> Float[Tensor, "V 3 B"]:
        return self.weights.shapedirs

    @property
    def posedirs(self) -> Float[Tensor, "P V*3"]:
        return self.weights.posedirs

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def skeleton_faces(self) -> Int[Tensor, "Fs 3"]:
        return self.weights.skel_faces

    @property
    def _feet_offset(self) -> Float[Tensor, "3"]:
        return self.weights.feet_offset

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Tensor, "B V 3"]:
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
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Tensor, "B 24 4 4"]:
        return backend.forward_skeleton(
            weights=self.weights,
            shape=shape,
            pose=body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
        )

    def forward_skeleton_mesh(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B Vs 3"]:
        _, _, _, _, skeleton_vertices = self._forward_full(shape, body_pose, global_rotation, global_translation)
        return skeleton_vertices + self.weights.feet_offset

    def forward_meshes(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B Vs 3"]]:
        vertices, _, _, _, skeleton_vertices = self._forward_full(shape, body_pose, global_rotation, global_translation)
        return vertices + self.weights.feet_offset, skeleton_vertices + self.weights.feet_offset

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.weights.v_template.device
        return {
            "shape": torch.zeros((1, self.NUM_BETAS), device=device, dtype=dtype),
            "body_pose": torch.zeros((batch_size, self.NUM_POSE_PARAMS), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    def get_tpose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        return self.get_rest_pose(batch_size=batch_size, **kwargs)

    def get_apose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, value in SKEL_APOSE.items():
            slices = (slice(None), index, 0) if body_pose.ndim == 3 else (slice(None), index)
            body_pose = common.set(body_pose, slices, value, xp=torch)
        params["body_pose"] = body_pose
        return params

    def get_ipose(
        self,
        batch_size: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        params = self.get_rest_pose(batch_size=batch_size, **kwargs)
        body_pose = params["body_pose"]
        for index, value in SKEL_IPOSE.items():
            slices = (slice(None), index, 0) if body_pose.ndim == 3 else (slice(None), index)
            body_pose = common.set(body_pose, slices, value, xp=torch)
        params["body_pose"] = body_pose
        return params

    def _forward_full(
        self,
        shape: Float[Tensor, "B 10"],
        body_pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> tuple[
        Float[Tensor, "B V 3"],
        Float[Tensor, "B 24 4 4"],
        Float[Tensor, "B 24 4 4"],
        Float[Tensor, "B 24 3"],
        Float[Tensor, "B Vs 3"],
    ]:
        weights = self.weights
        batch_shape = tuple(body_pose.shape[:-1])
        dtype = body_pose.dtype

        if global_translation is None:
            global_translation = torch.zeros((*batch_shape, 3), device=body_pose.device, dtype=dtype)
        if shape.shape[:-1] == (1,) and batch_shape:
            shape = torch.broadcast_to(shape, (*batch_shape, shape.shape[-1]))

        joints = weights.j_template + torch.einsum("jdi,...i->...jd", weights.j_shapedirs, shape)
        joint_rel = core._compute_J_rel(torch, joints, weights.parent)
        local_transforms = core._compute_local_transforms(
            torch,
            pose=body_pose,
            J=joints,
            J_rel=joint_rel,
            all_axes=weights.all_axes,
            rotation_indices=weights.rotation_indices,
            apose_R=weights.apose_R,
            apose_t=weights.apose_t,
            per_joint_rot=weights.per_joint_rot,
            child=weights.child,
            fixed_orientation_joints=weights.fixed_orientation_joints,
            scapula_r_axes=weights.scapula_r_axes,
            scapula_l_axes=weights.scapula_l_axes,
            spine_axes=weights.spine_axes,
        )
        transforms = core._propagate_transforms(torch, local_transforms, weights.parents[1:])

        v_shaped = weights.v_template + torch.einsum("vdi,...i->...vd", weights.shapedirs, shape)
        eye3 = torch.eye(3, device=body_pose.device, dtype=dtype)
        smpl_rotations = eye3.expand(*batch_shape, weights.num_joints_smpl, 3, 3).clone()
        smpl_rotations[..., core.SMPL_JOINT_MAP, :, :] = local_transforms[..., :, :3, :3]
        pose_feat = (smpl_rotations[..., 1:, :, :] - eye3).reshape(*batch_shape, -1)
        pose_offsets = (pose_feat @ weights.posedirs).reshape(*batch_shape, self.num_vertices, 3)
        v_posed = v_shaped + pose_offsets

        R_joint = transforms[..., :, :3, :3]
        t_world = transforms[..., :, :3, 3]
        t_skin = t_world - (R_joint @ joints[..., None]).squeeze(-1)
        W_R = torch.einsum("vj,...jkl->...vkl", weights.skin_weights, R_joint)
        W_t = torch.einsum("vj,...jk->...vk", weights.skin_weights, t_skin)
        vertices = (W_R @ v_posed[..., None]).squeeze(-1) + W_t

        Rk = core._compute_bone_orientation(
            torch,
            J_rel=joint_rel,
            apose_t=weights.apose_t,
            per_joint_rot=weights.per_joint_rot,
            child=weights.child,
            fixed_orientation_joints=weights.fixed_orientation_joints,
        )
        skeleton_vertices = self._compute_skeleton_vertices(joints, joint_rel, transforms, Rk, v_shaped)

        vertices = vertices + global_translation[..., None, :]
        skeleton_vertices = skeleton_vertices + global_translation[..., None, :]
        joints_out = transforms[..., :, :3, 3] + global_translation[..., None, :]

        if global_rotation is not None:
            rotation = SO3.conversions.from_axis_angle_to_rotmat(global_rotation, xp=torch)
            vertices = (rotation @ vertices.mT).mT
            skeleton_vertices = (rotation @ skeleton_vertices.mT).mT
            joints_out = (rotation @ joints_out.mT).mT

        return vertices, transforms, local_transforms, joints_out, skeleton_vertices

    def _compute_skeleton_vertices(
        self,
        joints: Float[Tensor, "B 24 3"],
        joint_rel: Float[Tensor, "B 24 3"],
        transforms: Float[Tensor, "B 24 4 4"],
        bone_orientation: Float[Tensor, "B 24 3 3"],
        v_shaped: Float[Tensor, "B V 3"],
    ) -> Float[Tensor, "B Vs 3"]:
        weights = self.weights
        device = joints.device
        dtype = joints.dtype
        batch_shape = tuple(joints.shape[:-2])
        num_joints = self.NUM_JOINTS

        apose_len = weights.apose_t.norm(dim=-1).expand(*batch_shape, -1)
        bone_len = joint_rel.norm(dim=-1)
        scale_ratio = bone_len / apose_len

        bone_scale = torch.ones(*batch_shape, num_joints, device=device, dtype=dtype)
        bone_scale[..., weights.non_leaf_joints] = scale_ratio[..., weights.non_leaf_children]
        bone_scale[..., [16, 21, 12]] = bone_scale[..., [17, 22, 11]]
        bone_scale[..., 11] = joint_rel[..., 11, 1].abs() / weights.apose_t[11, 1].abs()

        bone_scale = bone_scale.unsqueeze(-1).expand(*bone_scale.shape, 3).clone()
        for (joint, scale_axis, skin_axis), (v1, v2) in _SCALING_KEYPOINTS.items():
            scale = (v_shaped[..., v1, :] - v_shaped[..., v2, :]) / (weights.v_template[v1] - weights.v_template[v2])
            bone_scale[..., joint, scale_axis] = scale[..., skin_axis]

        s1 = (
            (v_shaped[..., 3027, :] - v_shaped[..., 3495, :]) / (weights.v_template[3027] - weights.v_template[3495])
        )[..., 2]
        s2 = (
            (v_shaped[..., 3027, :] - v_shaped[..., 3506, :]) / (weights.v_template[3027] - weights.v_template[3506])
        )[..., 2]
        bone_scale[..., 12, 0] = torch.min(s1, s2)
        bone_scale[..., 11, 0] = bone_scale[..., 12, 0]

        scale_matrices = torch.zeros(*batch_shape, num_joints, 4, 4, device=device, dtype=dtype)
        scale_matrices[..., :, 0, 0] = bone_scale[..., :, 0]
        scale_matrices[..., :, 1, 1] = bone_scale[..., :, 1]
        scale_matrices[..., :, 2, 2] = bone_scale[..., :, 2]
        scale_matrices[..., :, 3, 3] = 1

        aligned_transforms = _homog_matrix(bone_orientation, joints.unsqueeze(-1)) @ scale_matrices
        skel_h = torch.cat(
            [weights.skel_v_template, torch.ones(weights.skel_v_template.shape[0], 1, device=device)], -1
        )
        blend = torch.einsum("vj,...jxy->...vxy", weights.skel_weights_rigid, aligned_transforms)
        skel_aligned = (blend @ skel_h[..., None]).squeeze(-1)

        identity = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, num_joints, 3, 3)
        inverse_bind = _homog_matrix(identity, -joints.unsqueeze(-1))
        posed_transforms = transforms @ inverse_bind
        blend = torch.einsum("vj,...jxy->...vxy", weights.skel_weights, posed_transforms)
        return (blend @ skel_aligned[..., None]).squeeze(-1)[..., :, :3]


def _homog_matrix(R: Float[Tensor, "B J 3 3"], t: Float[Tensor, "B J 3 1"]) -> Float[Tensor, "B J 4 4"]:
    pad = R.new_tensor([0, 0, 0, 1]).expand(*R.shape[:-2], 1, 4)
    return torch.cat((torch.cat((R, t), -1), pad), -2)


_SCALING_KEYPOINTS = {
    (13, 0, 2): (410, 384),
    (13, 1, 1): (414, 384),
    (13, 2, 0): (196, 3708),
    (18, 0, 1): (6179, 6137),
    (18, 1, 0): (5670, 5906),
    (23, 0, 1): (6179, 6137),
    (23, 1, 0): (5670, 5906),
}
