"""PyTorch backend for SKEL model."""

import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from array_api_compat import get_namespace

from ..base import BodyModel
from . import core
from .io import get_model_path, simplify_mesh


class SKEL(BodyModel, nn.Module):
    """SKEL body model with anatomically realistic skeletal articulation.

    Args:
        gender: One of "male" or "female" (no neutral).
        model_path: Path to the SKEL model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.

    Note:
        forward_skeleton_mesh is PyTorch-only due to the complexity of bone scaling
        and skeleton mesh computation.
    """

    NUM_BETAS = 10
    NUM_JOINTS = 24
    NUM_POSE_PARAMS = 46

    # Buffer type annotations
    _v_template: Float[Tensor, "V 3"]
    _v_template_full: Float[Tensor, "V_full 3"]
    _faces: Int[Tensor, "F 3"]
    _skel_v_template: Float[Tensor, "Vs 3"]
    _skel_faces: Int[Tensor, "Fs 3"]
    J_regressor: Float[Tensor, "24 V_full"]
    _skin_weights: Float[Tensor, "V 24"]
    _skel_weights: Float[Tensor, "Vs 24"]
    _skel_weights_rigid: Float[Tensor, "Vs 24"]
    apose_R: Float[Tensor, "24 3 3"]
    apose_t: Float[Tensor, "24 3"]
    per_joint_rot: Float[Tensor, "24 3 3"]
    parent: Int[Tensor, "23"]
    child: Int[Tensor, "24"]
    fixed_orientation_joints: Int[Tensor, "6"]
    non_leaf_joints: Int[Tensor, "N"]
    non_leaf_children: Int[Tensor, "N"]
    _feet_offset: Float[Tensor, "3"]
    _all_axes: Float[Tensor, "47 3"]
    _rotation_indices: Int[Tensor, "24 3"]
    _scapula_r_axes: Float[Tensor, "3 3"]
    _scapula_l_axes: Float[Tensor, "3 3"]
    _spine_axes: Float[Tensor, "3 3"]

    def __init__(self, gender: str, model_path: Path | str | None = None, simplify: float = 1.0):
        assert gender in ("male", "female")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()
        self.gender = gender

        skel_file = get_model_path(model_path, gender)
        with open(skel_file, "rb") as f:
            data = pkl.load(f)

        self._init_buffers(data, simplify)
        self._init_joints()

    def _init_buffers(self, data: dict, simplify: float) -> None:
        # Load full-resolution data
        v_template_full = np.asarray(data["skin_template_v"], dtype=np.float32)
        faces = np.asarray(data["skin_template_f"], dtype=np.int32)
        shapedirs_full = np.asarray(data["shapedirs"][:, :, : self.NUM_BETAS], dtype=np.float32)
        posedirs_full = np.asarray(data["posedirs"], dtype=np.float32)
        skin_weights = _sparse_to_dense(data["skin_weights"]).numpy()

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs_full[vertex_map]
            skin_weights = skin_weights[vertex_map]
        else:
            v_template = v_template_full
            shapedirs = shapedirs_full
            posedirs = posedirs_full

        # Full-resolution buffers for skeleton computation
        self.register_buffer("_v_template_full", torch.as_tensor(v_template_full))
        self.shapedirs_full = nn.Parameter(torch.as_tensor(shapedirs_full), requires_grad=False)
        self.posedirs_full = nn.Parameter(torch.as_tensor(posedirs_full), requires_grad=False)
        self.register_buffer("J_regressor", _sparse_to_dense(data["J_regressor_osim"]))

        # Simplified buffers for mesh output
        self.register_buffer("_v_template", torch.as_tensor(v_template))
        self.register_buffer("_faces", torch.as_tensor(faces, dtype=torch.int64))
        self.shapedirs = nn.Parameter(torch.as_tensor(shapedirs), requires_grad=False)
        self.posedirs = nn.Parameter(torch.as_tensor(posedirs), requires_grad=False)
        self.register_buffer("_skin_weights", torch.as_tensor(skin_weights))

        # Skeleton mesh (not simplified - separate mesh)
        self.register_buffer("_skel_v_template", torch.tensor(data["skel_template_v"], dtype=torch.float32))
        self.register_buffer("_skel_faces", torch.tensor(data["skel_template_f"], dtype=torch.int64))
        self.register_buffer("_skel_weights", _sparse_to_dense(data["skel_weights"]))
        self.register_buffer("_skel_weights_rigid", _sparse_to_dense(data["skel_weights_rigid"]))

        # Anatomical pose data
        apose = torch.tensor(data["apose_rel_transfo"], dtype=torch.float32)
        self.register_buffer("apose_R", apose[:, :3, :3])
        self.register_buffer("apose_t", apose[:, :3, 3])
        self.register_buffer("per_joint_rot", torch.tensor(data["per_joint_rot"], dtype=torch.float32))

        # Kinematic tree
        kintree = torch.tensor(data["osim_kintree_table"], dtype=torch.int64)
        id_to_col = {kintree[1, i].item(): i for i in range(kintree.shape[1])}
        self.parent_list = [id_to_col[kintree[0, i].item()] for i in range(1, kintree.shape[1])]
        self.register_buffer("parent", torch.tensor(self.parent_list, dtype=torch.int64))

        # Child indices for bone orientation
        child_list = []
        for i in range(self.NUM_JOINTS):
            children = (kintree[0] == i).nonzero(as_tuple=True)[0]
            child_list.append(kintree[1, children[0]].item() if len(children) > 0 else 0)
        self.register_buffer("child", torch.tensor(child_list, dtype=torch.int64))

        # Indices where bone orientation is fixed (pelvis, feet, head)
        self.register_buffer("fixed_orientation_joints", torch.tensor([0, 5, 10, 13, 18, 23], dtype=torch.int64))

        # Non-leaf joints for bone scaling
        non_leaf = [i for i, c in enumerate(child_list) if c != 0]
        self.register_buffer("non_leaf_joints", torch.tensor(non_leaf, dtype=torch.int64))
        self.register_buffer("non_leaf_children", torch.tensor([child_list[i] for i in non_leaf], dtype=torch.int64))

        # Number of SMPL joints (for pose blend shapes)
        self.num_joints_smpl = data["J_regressor"].shape[0]

        # Feet offset for Y=0 floor
        y_offset = -v_template[:, 1].min().item()
        self.register_buffer("_feet_offset", torch.tensor([0.0, y_offset, 0.0], dtype=torch.float32))

    def _init_joints(self) -> None:
        """Precompute axes and indices for vectorized joint rotation computation."""
        from nanomanifold import SO3

        def pin_axis_from_euler(euler_xyz):
            """Compute pin joint axis from euler angles (XYZ convention)."""
            euler = torch.tensor(euler_xyz, dtype=torch.float32)
            R = SO3.to_matrix(SO3.from_euler(euler, convention="XYZ"))
            axis = R @ torch.tensor([0.0, 0.0, 1.0])
            return axis.tolist()

        # Joint axes based on joint types
        # Pin joints use euler angles that get converted to axes via SO3 rotation
        joint_axes = [
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # pelvis - 3 DOF
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],  # femur_r - 3 DOF
            [[0, 0, -1]],  # tibia_r (knee) - 1 DOF
            [pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],  # talus_r - 1 DOF
            [pin_axis_from_euler([-1.76818999, 0.906223, 1.8196])],  # calcn_r - 1 DOF
            [pin_axis_from_euler([-3.14158999, 0.619901, 0])],  # toes_r - 1 DOF
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],  # femur_l - 3 DOF
            [[0, 0, -1]],  # tibia_l (knee) - 1 DOF
            [pin_axis_from_euler([0.175895, -0.105208, 0.0186622])],  # talus_l - 1 DOF
            [pin_axis_from_euler([1.76818999, -0.906223, 1.8196])],  # calcn_l - 1 DOF
            [pin_axis_from_euler([-3.14158999, -0.619901, 0])],  # toes_l - 1 DOF
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # lumbar_body - 3 DOF
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # thorax - 3 DOF
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],  # head - 3 DOF
            [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],  # scapula_r - 3 DOF
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # humerus_r - 3 DOF
            [[0.0494, 0.0366, 0.99810825]],  # ulna_r - 1 DOF (MultiAxisJoint, not PinJoint)
            [[-0.01716099, 0.99266564, -0.11966796]],  # radius_r - 1 DOF (MultiAxisJoint)
            [[1, 0, 0], [0, 0, -1]],  # hand_r - 2 DOF
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # scapula_l - 3 DOF
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # humerus_l - 3 DOF
            [[-0.0494, -0.0366, 0.99810825]],  # ulna_l - 1 DOF (MultiAxisJoint)
            [[0.01716099, -0.99266564, -0.11966796]],  # radius_l - 1 DOF (MultiAxisJoint)
            [[-1, 0, 0], [0, 0, -1]],  # hand_l - 2 DOF
        ]

        # Precompute axes and indices
        all_axes: list[list[float]] = []
        rotation_indices: list[list[int]] = []
        dof_idx = 0

        for joint_idx, axes in enumerate(joint_axes):
            n_dof = len(axes)

            # Collect axes for this joint
            for ax in axes:
                ax_arr = np.array(ax, dtype=np.float32)
                # Only normalize multi-axis joint axes (pin joint axes are already normalized)
                ax_norm = np.linalg.norm(ax_arr)
                if ax_norm > 1e-6 and abs(ax_norm - 1.0) > 1e-6:
                    ax_arr = ax_arr / ax_norm
                all_axes.append(ax_arr.tolist())

            # Build rotation indices: [idx0, idx1, idx2] with identity padding
            identity_idx = self.NUM_POSE_PARAMS
            if n_dof == 1:
                rotation_indices.append([dof_idx, identity_idx, identity_idx])
            elif n_dof == 2:
                rotation_indices.append([dof_idx, dof_idx + 1, identity_idx])
            else:
                rotation_indices.append([dof_idx, dof_idx + 1, dof_idx + 2])

            dof_idx += n_dof

        # Append zero axis (produces identity rotation)
        all_axes.append([0.0, 0.0, 0.0])

        self.register_buffer("_all_axes", torch.tensor(all_axes, dtype=torch.float32))
        self.register_buffer("_rotation_indices", torch.tensor(rotation_indices, dtype=torch.long))

        # Store joint axes for scapula and spine (needed for offset computation)
        self.register_buffer("_scapula_r_axes", torch.tensor([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=torch.float32))
        self.register_buffer("_scapula_l_axes", torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32))
        self.register_buffer("_spine_axes", torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self._v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 24"]:
        return self._skin_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self._v_template + self._feet_offset

    @property
    def skeleton_faces(self) -> Int[Tensor, "Fs 3"]:
        """Skeleton mesh face indices."""
        return self._skel_faces

    # -------------------------------------------------------------------------
    # Forward API
    # -------------------------------------------------------------------------

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        return core.forward_vertices(
            v_template=self._v_template,
            v_template_full=self._v_template_full,
            shapedirs=self.shapedirs,
            shapedirs_full=self.shapedirs_full,
            posedirs=self.posedirs.view(-1, self.posedirs.shape[-1]).T,
            skin_weights=self._skin_weights,
            J_regressor=self.J_regressor,
            parents=self.parent,
            all_axes=self._all_axes,
            rotation_indices=self._rotation_indices,
            apose_R=self.apose_R,
            apose_t=self.apose_t,
            per_joint_rot=self.per_joint_rot,
            child=self.child,
            fixed_orientation_joints=self.fixed_orientation_joints,
            feet_offset=self._feet_offset,
            num_joints_smpl=self.num_joints_smpl,
            scapula_r_axes=self._scapula_r_axes,
            scapula_l_axes=self._scapula_l_axes,
            spine_axes=self._spine_axes,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            xp=torch,
        )

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        """Compute skeleton joint transforms [B, 24, 4, 4]."""
        return core.forward_skeleton(
            v_template_full=self._v_template_full,
            shapedirs_full=self.shapedirs_full,
            J_regressor=self.J_regressor,
            parents=self.parent,
            all_axes=self._all_axes,
            rotation_indices=self._rotation_indices,
            apose_R=self.apose_R,
            apose_t=self.apose_t,
            per_joint_rot=self.per_joint_rot,
            child=self.child,
            fixed_orientation_joints=self.fixed_orientation_joints,
            feet_offset=self._feet_offset,
            scapula_r_axes=self._scapula_r_axes,
            scapula_l_axes=self._scapula_l_axes,
            spine_axes=self._spine_axes,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            xp=torch,
        )

    def forward_skeleton_mesh(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B Vs 3"]:
        """Compute skeleton mesh vertices [B, Vs, 3].

        Note: This is PyTorch-only due to the complexity of bone scaling computation.
        """
        _, _, _, _, skel_v = self._forward_full(shape, pose, global_rotation, global_translation)
        return skel_v + self._feet_offset

    def forward_meshes(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B Vs 3"]]:
        """Compute both skin and skeleton mesh vertices efficiently.

        Note: This is PyTorch-only due to the skeleton mesh computation.

        Returns:
            vertices: Skin mesh vertices [B, V, 3]
            skeleton_vertices: Skeleton mesh vertices [B, Vs, 3]
        """
        v, _, _, _, skel_v = self._forward_full(shape, pose, global_rotation, global_translation)
        return v + self._feet_offset, skel_v + self._feet_offset

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self._v_template.device
        return {
            "shape": torch.zeros((1, self.NUM_BETAS), device=device, dtype=dtype),
            "pose": torch.zeros((batch_size, self.NUM_POSE_PARAMS), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    # -------------------------------------------------------------------------
    # PyTorch-only: Full forward with skeleton mesh
    # -------------------------------------------------------------------------

    def _forward_full(
        self,
        shape: Float[Tensor, "B 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> tuple[
        Float[Tensor, "B V 3"],
        Float[Tensor, "B 24 4 4"],
        Float[Tensor, "B 24 4 4"],
        Float[Tensor, "B 24 3"],
        Float[Tensor, "B Vs 3"],
    ]:
        """Full forward: vertices, transforms, joints, skeleton vertices (PyTorch only)."""
        xp = get_namespace(pose)
        B = pose.shape[0]
        device, dtype = pose.device, pose.dtype
        Nv = self._v_template.shape[0]

        if global_translation is None:
            global_translation = torch.zeros((B, 3), device=device, dtype=dtype)
        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)

        # Joint positions (use full-resolution for accurate skeleton)
        v_shaped_full = self._v_template_full + torch.einsum("vdi,bi->bvd", self.shapedirs_full, shape)
        J = torch.einsum("bvd,jv->bjd", v_shaped_full, self.J_regressor)
        J_rel = core._compute_J_rel(xp, J, self.parent)

        # Forward kinematics using core functions
        G_local = core._compute_local_transforms(
            xp,
            pose=pose,
            J=J,
            J_rel=J_rel,
            all_axes=self._all_axes,
            rotation_indices=self._rotation_indices,
            apose_R=self.apose_R,
            apose_t=self.apose_t,
            per_joint_rot=self.per_joint_rot,
            child=self.child,
            fixed_orientation_joints=self.fixed_orientation_joints,
            scapula_r_axes=self._scapula_r_axes,
            scapula_l_axes=self._scapula_l_axes,
            spine_axes=self._spine_axes,
        )
        G = core._propagate_transforms(xp, G_local, self.parent)

        # Shape blend shapes (simplified mesh for output)
        v_shaped = self._v_template + torch.einsum("vdi,bi->bvd", self.shapedirs, shape)

        # Pose blend shapes (SMPL-compatible)
        R_smpl = _batch_eye(B, self.num_joints_smpl, 3, device, dtype).clone()
        R_smpl[:, core.SMPL_JOINT_MAP] = G_local[:, :, :3, :3]
        pose_feat = (R_smpl[:, 1:] - torch.eye(3, device=device, dtype=dtype)).view(B, -1)
        pose_offsets = (pose_feat @ self.posedirs.view(Nv * 3, -1).T).view(B, Nv, 3)
        v_posed = v_shaped + pose_offsets

        # Skin LBS
        R_joint = G[:, :, :3, :3]
        t_world = G[:, :, :3, 3]
        t_skin = t_world - (R_joint @ J[..., None]).squeeze(-1)

        W_R = torch.einsum("vj,bjkl->bvkl", self._skin_weights, R_joint)
        W_t = torch.einsum("vj,bjk->bvk", self._skin_weights, t_skin)
        v_out = (W_R @ v_posed[..., None]).squeeze(-1) + W_t

        # Skeleton mesh (PyTorch-only)
        Rk = core._compute_bone_orientation(
            xp,
            J_rel=J_rel,
            apose_t=self.apose_t,
            per_joint_rot=self.per_joint_rot,
            child=self.child,
            fixed_orientation_joints=self.fixed_orientation_joints,
        )
        skel_v = self._compute_skeleton_vertices(B, J, J_rel, G, Rk, v_shaped)

        # Apply global transform
        v_out = v_out + global_translation[:, None]
        skel_v = skel_v + global_translation[:, None]
        J_out = G[:, :, :3, 3] + global_translation[:, None]

        if global_rotation is not None:
            R = _axisangle_to_matrix(global_rotation)
            v_out = (R @ v_out.mT).mT
            skel_v = (R @ skel_v.mT).mT
            J_out = (R @ J_out.mT).mT

        return v_out, G, G_local, J_out, skel_v

    def _compute_skeleton_vertices(
        self,
        B: int,
        J: Float[Tensor, "B 24 3"],
        J_rel: Float[Tensor, "B 24 3"],
        G: Float[Tensor, "B 24 4 4"],
        Rk: Float[Tensor, "B 24 3 3"],
        v_shaped: Float[Tensor, "B V 3"],
    ) -> Float[Tensor, "B Vs 3"]:
        """Compute skeleton mesh vertices (PyTorch only)."""
        device, dtype = J.device, J.dtype
        Nj = self.NUM_JOINTS

        # Bone scaling
        apose_len = self.apose_t.norm(dim=-1).unsqueeze(0).expand(B, -1)
        bone_len = J_rel.norm(dim=-1)
        scale_ratio = bone_len / apose_len

        bone_scale = torch.ones(B, Nj, device=device, dtype=dtype)
        bone_scale[:, self.non_leaf_joints] = scale_ratio[:, self.non_leaf_children]
        bone_scale[:, [16, 21, 12]] = bone_scale[:, [17, 22, 11]]
        bone_scale[:, 11] = J_rel[:, 11, 1].abs() / self.apose_t[11, 1].abs()

        bone_scale = bone_scale.unsqueeze(-1).expand(-1, -1, 3).clone()
        v0 = self._v_template.unsqueeze(0)
        for (ji, doi, dsi), (v1, v2) in _SCALING_KEYPOINTS.items():
            bone_scale[:, ji, doi] = ((v_shaped[:, v1] - v_shaped[:, v2]) / (v0[:, v1] - v0[:, v2]))[:, dsi]

        # Thorax special case
        s1 = ((v_shaped[:, 3027] - v_shaped[:, 3495]) / (v0[:, 3027] - v0[:, 3495]))[:, 2]
        s2 = ((v_shaped[:, 3027] - v_shaped[:, 3506]) / (v0[:, 3027] - v0[:, 3506]))[:, 2]
        bone_scale[:, 12, 0] = torch.min(s1, s2)
        bone_scale[:, 11, 0] = bone_scale[:, 12, 0]

        # Build scale matrices and apply
        S = torch.zeros(B, Nj, 4, 4, device=device, dtype=dtype)
        S[:, :, 0, 0] = bone_scale[:, :, 0]
        S[:, :, 1, 1] = bone_scale[:, :, 1]
        S[:, :, 2, 2] = bone_scale[:, :, 2]
        S[:, :, 3, 3] = 1

        Gk_mat = _homog_matrix(Rk, J.unsqueeze(-1)) @ S
        skel_v_h = torch.cat([self._skel_v_template, torch.ones(self._skel_v_template.shape[0], 1, device=device)], -1)
        T = torch.einsum("vj,bjxy->bvxy", self._skel_weights_rigid, Gk_mat)
        skel_aligned = (T @ skel_v_h.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

        # Transform to posed space
        G_inv = _homog_matrix(_batch_eye(B, Nj, 3, device, dtype), -J.unsqueeze(-1))
        G_posed = G @ G_inv
        T = torch.einsum("vj,bjxy->bvxy", self._skel_weights, G_posed)
        return (T @ skel_aligned.unsqueeze(-1)).squeeze(-1)[:, :, :3]


# =============================================================================
# Utilities (PyTorch-specific)
# =============================================================================


def _axisangle_to_matrix(v: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3 3"]:
    """Rodrigues formula: axis-angle [*, 3] -> rotation matrix [*, 3, 3]."""
    theta = torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    r = v / theta
    rx, ry, rz = r.unbind(-1)
    z = torch.zeros_like(rx)
    K = torch.stack([torch.stack([z, -rz, ry], -1), torch.stack([rz, z, -rx], -1), torch.stack([-ry, rx, z], -1)], -2)
    sin_t, cos_t = torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)
    return torch.eye(3, device=v.device, dtype=v.dtype) + sin_t * K + (1 - cos_t) * (K @ K)


def _sparse_to_dense(arr_coo) -> Tensor:
    indices = torch.tensor(np.vstack((arr_coo.row, arr_coo.col)), dtype=torch.int64)
    values = torch.tensor(arr_coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, arr_coo.shape).to_dense()


def _homog_matrix(R: Float[Tensor, "B J 3 3"], t: Float[Tensor, "B J 3 1"]) -> Float[Tensor, "B J 4 4"]:
    """Build [B, J, 4, 4] homogeneous matrix from rotation and translation."""
    pad = R.new_tensor([0, 0, 0, 1]).expand(*R.shape[:2], 1, 4)
    return torch.cat((torch.cat((R, t), -1), pad), 2)


def _batch_eye(B: int, N: int, d: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "B N d d"]:
    return torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d).expand(B, N, d, d)


def _skew(v: Float[Tensor, "B N 3"]) -> Float[Tensor, "B N 3 3"]:
    """Skew-symmetric matrix from vector."""
    z = torch.zeros_like(v[..., :1])
    return torch.stack(
        [
            torch.cat([z, -v[..., 2:3], v[..., 1:2]], -1),
            torch.cat([v[..., 2:3], z, -v[..., 0:1]], -1),
            torch.cat([-v[..., 1:2], v[..., 0:1], z], -1),
        ],
        -2,
    )


def _rotation_between_vectors(a: Float[Tensor, "B N 3"], b: Float[Tensor, "B N 3"]) -> Float[Tensor, "B N 3 3"]:
    """Rotation matrix that rotates normalized vectors a to b."""
    a = a / torch.linalg.norm(a, dim=-1, keepdim=True)
    b = b / torch.linalg.norm(b, dim=-1, keepdim=True)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    s = torch.linalg.norm(v, dim=-1) + 1e-7

    K = _skew(v)
    I = _batch_eye(a.shape[0], a.shape[1], 3, a.device, a.dtype)
    scale = ((1 - c) / (s**2)).view(*a.shape[:2], 1, 1)
    return I + K + (K @ K) * scale


# =============================================================================
# Constants
# =============================================================================

# Vertex pairs for anisotropic bone scaling: (joint, axis, skin_axis) -> (v1, v2)
_SCALING_KEYPOINTS = {
    (13, 0, 2): (410, 384),
    (13, 1, 1): (414, 384),
    (13, 2, 0): (196, 3708),
    (18, 0, 1): (6179, 6137),
    (18, 1, 0): (5670, 5906),
    (23, 0, 1): (6179, 6137),
    (23, 1, 0): (5670, 5906),
}


# =============================================================================
# Conversion functions
# =============================================================================


def from_native_args(
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 46"],
    root_rotation: Float[Tensor, "B 3"] | None = None,
    global_rotation: Float[Tensor, "B 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
) -> dict[str, Tensor | None]:
    """Convert native SKEL args to forward_* kwargs."""
    return core.from_native_args(shape, body_pose, root_rotation, global_rotation, global_translation)


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    skeleton_mesh: Float[Tensor, "B Vs 3"],
    feet_offset: Float[Tensor, "3"],
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native SKEL format."""
    result = core.to_native_outputs(vertices, transforms, feet_offset)
    result["skeleton_vertices"] = skeleton_mesh - feet_offset
    return result
