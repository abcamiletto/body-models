import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from .io import get_model_path


class SKEL(BodyModel):
    """SKEL body model with anatomically realistic skeletal articulation.

    Args:
        gender: One of "male" or "female" (no neutral).
        model_path: Path to the SKEL model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
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
            v_template, faces, vertex_map = _simplify_mesh(v_template_full, faces, target_faces)
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
        y_offset = -self._v_template[:, 1].min().item()
        self.register_buffer("_feet_offset", torch.tensor([0.0, y_offset, 0.0], dtype=torch.float32))

    def _init_joints(self) -> None:
        # Joint order must match pose parameter layout (46 total DOFs)
        # Joints with offset methods (scapula, spine) are stored separately for type safety
        self._scapula_r = _ScapulaJoint([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], is_left=False)
        self._scapula_l = _ScapulaJoint([[0, 1, 0], [0, 0, 1], [1, 0, 0]], is_left=True)
        self._lumbar = _SpineJoint([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self._thorax = _SpineJoint([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self._head = _SpineJoint([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        self.joints = nn.ModuleList(
            [
                _MultiAxisJoint([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),  # pelvis
                _MultiAxisJoint([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),  # femur_r
                _KneeJoint(),  # tibia_r
                _PinJoint([0.175895, -0.105208, 0.0186622]),  # talus_r
                _PinJoint([-1.76818999, 0.906223, 1.8196]),  # calcn_r
                _PinJoint([-3.14158999, 0.619901, 0]),  # toes_r
                _MultiAxisJoint([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),  # femur_l
                _KneeJoint(),  # tibia_l
                _PinJoint([0.175895, -0.105208, 0.0186622]),  # talus_l
                _PinJoint([1.76818999, -0.906223, 1.8196]),  # calcn_l
                _PinJoint([-3.14158999, -0.619901, 0]),  # toes_l
                self._lumbar,  # lumbar_body
                self._thorax,  # thorax
                self._head,  # head
                self._scapula_r,  # scapula_r
                _MultiAxisJoint([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # humerus_r
                _MultiAxisJoint([[0.0494, 0.0366, 0.99810825]]),  # ulna_r
                _MultiAxisJoint([[-0.01716099, 0.99266564, -0.11966796]]),  # radius_r
                _MultiAxisJoint([[1, 0, 0], [0, 0, -1]]),  # hand_r
                self._scapula_l,  # scapula_l
                _MultiAxisJoint([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # humerus_l
                _MultiAxisJoint([[-0.0494, -0.0366, 0.99810825]]),  # ulna_l
                _MultiAxisJoint([[0.01716099, -0.99266564, -0.11966796]]),  # radius_l
                _MultiAxisJoint([[-1, 0, 0], [0, 0, -1]]),  # hand_l
            ]
        )

        # Precompute axes and indices for batched joint rotations
        self._precompute_joint_axes()

    def _precompute_joint_axes(self) -> None:
        """Precompute axes and indices for vectorized joint rotation computation."""
        all_axes: list[list[float]] = []
        rotation_indices: list[list[int]] = []
        dof_idx = 0

        for joint in self.joints:
            n_dof: int = joint.n_dof  # type: ignore[assignment]

            # Collect axes for this joint
            if type(joint).__name__ == "_KneeJoint":
                all_axes.append([0.0, 0.0, -1.0])
            elif hasattr(joint, "axes"):
                axes: Tensor = joint.axes  # type: ignore[assignment]
                for ax in axes:
                    all_axes.append(ax.tolist())
            elif hasattr(joint, "axis"):
                axis: Tensor = joint.axis
                all_axes.append(axis.tolist())

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

    # -------------------------------------------------------------------------
    # Forward API
    # -------------------------------------------------------------------------

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        expression: Float[Tensor, "B _"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        v = self._forward_vertices_only(shape, pose, global_rotation, global_translation)
        return v + self._feet_offset

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        expression: Float[Tensor, "B _"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        G = self._forward_skeleton_only(shape, pose, global_rotation, global_translation)
        G = G.clone()
        G[..., :3, 3] = G[..., :3, 3] + self._feet_offset
        return G

    def forward_skeleton_mesh(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        expression: Float[Tensor, "B _"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B Vs 3"]:
        """Compute skeleton mesh vertices [B, Vs, 3]."""
        _, _, _, _, skel_v = self._forward(shape, pose, global_rotation, global_translation)
        return skel_v + self._feet_offset

    def forward_meshes(
        self,
        shape: Float[Tensor, "B|1 10"],
        pose: Float[Tensor, "B 46"],
        expression: Float[Tensor, "B _"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B Vs 3"]]:
        """Compute both skin and skeleton mesh vertices efficiently.

        Returns:
            vertices: Skin mesh vertices [B, 6890, 3]
            skeleton_vertices: Skeleton mesh vertices [B, 247252, 3]
        """
        v, _, _, _, skel_v = self._forward(shape, pose, global_rotation, global_translation)
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
    # Core forward passes
    # -------------------------------------------------------------------------

    def _forward_skeleton_only(
        self,
        shape: Float[Tensor, "B 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        """Fast path: compute only joint transforms, skip all vertex operations."""
        B = pose.shape[0]
        device, dtype = pose.device, pose.dtype

        if global_translation is None:
            global_translation = torch.zeros((B, 3), device=device, dtype=dtype)
        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)

        # Shape blend shapes -> joint positions (use full-resolution for accurate skeleton)
        v_shaped_full = self._v_template_full + torch.einsum("vdi,bi->bvd", self.shapedirs_full, shape)
        J = torch.einsum("bvd,jv->bjd", v_shaped_full, self.J_regressor)
        J_rel = torch.cat([J[:, :1], J[:, 1:] - J[:, self.parent]], dim=1)

        # Forward kinematics
        G_local = self._compute_local_transforms(pose, J, J_rel)
        G = self._propagate_transforms(G_local)

        # Apply global transform
        rot = G[:, :, :3, :3]
        trans = G[:, :, :3, 3]
        if global_rotation is not None:
            R = _axisangle_to_matrix(global_rotation)
            rot = R[:, None] @ rot
            trans = (R @ trans.mT).mT
        trans = trans + global_translation[:, None]

        last_row = G.new_tensor([0, 0, 0, 1]).expand(B, self.NUM_JOINTS, 1, 4)
        G = torch.cat([torch.cat([rot, trans.unsqueeze(-1)], dim=-1), last_row], dim=2)
        return G

    def _forward_vertices_only(
        self,
        shape: Float[Tensor, "B 10"],
        pose: Float[Tensor, "B 46"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Fast path for vertices: skip skeleton mesh computation."""
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
        J_rel = J.clone()
        J_rel[:, 1:] -= J[:, self.parent]

        # Forward kinematics
        G_local = self._compute_local_transforms(pose, J, J_rel)
        G = self._propagate_transforms(G_local)

        # Shape blend shapes (simplified mesh for output)
        v_shaped = self._v_template + torch.einsum("vdi,bi->bvd", self.shapedirs, shape)

        # Pose blend shapes (SMPL-compatible)
        R_smpl = _batch_eye(B, self.num_joints_smpl, 3, device, dtype).clone()
        R_smpl[:, _SMPL_JOINT_MAP] = G_local[:, :, :3, :3]
        pose_feat = (R_smpl[:, 1:] - torch.eye(3, device=device, dtype=dtype)).view(B, -1)
        pose_offsets = (pose_feat @ self.posedirs.view(Nv * 3, -1).T).view(B, Nv, 3)
        v_posed = v_shaped + pose_offsets

        # Skin LBS (optimized: separate R and t, avoid homogeneous coordinates)
        R_joint = G[:, :, :3, :3]  # [B, J, 3, 3]
        t_world = G[:, :, :3, 3]  # [B, J, 3]
        t_skin = t_world - (R_joint @ J[..., None]).squeeze(-1)  # [B, J, 3]

        W_R = torch.einsum("vj,bjkl->bvkl", self._skin_weights, R_joint)  # [B, V, 3, 3]
        W_t = torch.einsum("vj,bjk->bvk", self._skin_weights, t_skin)  # [B, V, 3]
        v_out = (W_R @ v_posed[..., None]).squeeze(-1) + W_t

        # Apply global transform
        v_out = v_out + global_translation[:, None]
        if global_rotation is not None:
            R = _axisangle_to_matrix(global_rotation)
            v_out = (R @ v_out.mT).mT

        return v_out

    def _forward(
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
        """Full forward: vertices, transforms, joints, skeleton vertices."""
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
        J_rel = torch.cat([J[:, :1], J[:, 1:] - J[:, self.parent]], dim=1)

        # Forward kinematics
        G_local = self._compute_local_transforms(pose, J, J_rel)
        G = self._propagate_transforms(G_local)

        # Shape blend shapes (simplified mesh for output)
        v_shaped = self._v_template + torch.einsum("vdi,bi->bvd", self.shapedirs, shape)

        # Pose blend shapes (SMPL-compatible)
        R_smpl = _batch_eye(B, self.num_joints_smpl, 3, device, dtype).clone()
        R_smpl[:, _SMPL_JOINT_MAP] = G_local[:, :, :3, :3]
        pose_feat = (R_smpl[:, 1:] - torch.eye(3, device=device, dtype=dtype)).view(B, -1)
        pose_offsets = (pose_feat @ self.posedirs.view(Nv * 3, -1).T).view(B, Nv, 3)
        v_posed = v_shaped + pose_offsets

        # Skin LBS (optimized: separate R and t, avoid homogeneous coordinates)
        R_joint = G[:, :, :3, :3]  # [B, J, 3, 3]
        t_world = G[:, :, :3, 3]  # [B, J, 3]
        t_skin = t_world - (R_joint @ J[..., None]).squeeze(-1)  # [B, J, 3]

        W_R = torch.einsum("vj,bjkl->bvkl", self._skin_weights, R_joint)  # [B, V, 3, 3]
        W_t = torch.einsum("vj,bjk->bvk", self._skin_weights, t_skin)  # [B, V, 3]
        v_out = (W_R @ v_posed[..., None]).squeeze(-1) + W_t

        # Skeleton mesh
        skel_v = self._compute_skeleton_vertices(B, J, J_rel, G, self._compute_bone_orientation(J_rel), v_shaped)

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

    def _compute_local_transforms(
        self,
        pose: Float[Tensor, "B 46"],
        J: Float[Tensor, "B 24 3"],
        J_rel: Float[Tensor, "B 24 3"],
    ) -> Float[Tensor, "B 24 4 4"]:
        """Compute local joint transforms from pose parameters."""
        B = pose.shape[0]

        # Bone orientation correction
        Rk = self._compute_bone_orientation(J_rel)
        Ra = self.apose_R.unsqueeze(0).expand(B, -1, -1, -1)

        # Batched joint rotations: convert all axis-angles to matrices at once
        # Pad pose with zero for identity rotation (used by joints with < 3 DOFs)
        pose_padded = torch.cat([pose, torch.zeros(B, 1, device=pose.device, dtype=pose.dtype)], dim=1)
        axis_angles = pose_padded.unsqueeze(-1) * self._all_axes  # [B, 47, 3]
        all_R = _axisangle_to_matrix(axis_angles)  # [B, 47, 3, 3]

        # Compose rotations: Rp = R2 @ R1 @ R0 (identity-padded for joints with fewer DOFs)
        R0 = all_R[:, self._rotation_indices[:, 0]]  # [B, J, 3, 3]
        R1 = all_R[:, self._rotation_indices[:, 1]]
        R2 = all_R[:, self._rotation_indices[:, 2]]
        Rp = R2 @ (R1 @ R0)

        # Compose rotations
        R = Rk @ (Ra.mT @ (Rp @ (Ra @ Rk.mT)))

        # Translation with anatomical adjustments
        t_base = J_rel.unsqueeze(-1)

        # Scapula adjustments
        thorax_w = (J[:, 19] - J[:, 14]).norm(dim=1)
        thorax_h = (J[:, 12] - J[:, 11]).norm(dim=1)
        offset_r = self._scapula_r.offset(pose[:, 26], pose[:, 27], thorax_w, thorax_h).unsqueeze(1)
        offset_l = self._scapula_l.offset(pose[:, 36], pose[:, 37], thorax_w, thorax_h).unsqueeze(1)

        # Spine adjustments
        offset_11 = self._lumbar.offset(pose[:, 17], pose[:, 18], (J[:, 11, 1] - J[:, 0, 1]).abs()).unsqueeze(1)
        offset_12 = self._thorax.offset(pose[:, 20], pose[:, 21], (J[:, 12, 1] - J[:, 11, 1]).abs()).unsqueeze(1)
        offset_13 = self._head.offset(pose[:, 23], pose[:, 24], (J[:, 13, 1] - J[:, 12, 1]).abs()).unsqueeze(1)

        zero = torch.zeros_like(t_base[:, :1])
        offsets = [zero for _ in range(self.NUM_JOINTS)]
        offsets[14] = offset_r
        offsets[19] = offset_l
        offsets[11] = offset_11
        offsets[12] = offset_12
        offsets[13] = offset_13
        t = t_base + torch.cat(offsets, dim=1)

        return _homog_matrix(R, t)

    def _propagate_transforms(self, G_local: Float[Tensor, "B 24 4 4"]) -> Float[Tensor, "B 24 4 4"]:
        """Propagate local transforms to world space."""
        G_list = [G_local[:, 0]]
        for i in range(1, self.NUM_JOINTS):
            G_list.append(G_list[self.parent_list[i - 1]] @ G_local[:, i])
        return torch.stack(G_list, dim=1)

    def _compute_bone_orientation(self, J_rel: Float[Tensor, "B 24 3"]) -> Float[Tensor, "B 24 3 3"]:
        """Compute per-joint orientation corrections."""
        B = J_rel.shape[0]

        bone_vec = J_rel[:, self.child]
        bone_vec_parts = [bone_vec[:, i : i + 1] for i in range(self.NUM_JOINTS)]
        bone_vec_parts[16] = bone_vec[:, 16:17] + bone_vec[:, 17:18]
        bone_vec_parts[21] = bone_vec[:, 21:22] + bone_vec[:, 22:23]
        bone_vec_parts[12] = bone_vec[:, 11:12]
        bone_vec = torch.cat(bone_vec_parts, dim=1)

        apose_vec = self.apose_t[self.child].unsqueeze(0).expand(B, -1, -1)
        apose_parts = [apose_vec[:, i : i + 1] for i in range(self.NUM_JOINTS)]
        apose_parts[16] = apose_vec[:, 16:17] + apose_vec[:, 17:18]
        apose_parts[21] = apose_vec[:, 21:22] + apose_vec[:, 22:23]
        apose_vec = torch.cat(apose_parts, dim=1)

        Gk_learned = self.per_joint_rot.unsqueeze(0).expand(B, -1, -1, -1)
        apose_corrected = (Gk_learned @ apose_vec.unsqueeze(-1)).squeeze(-1)

        Gk = _rotation_between_vectors(apose_corrected, bone_vec)
        Gk = torch.where(torch.isnan(Gk), torch.zeros_like(Gk), Gk)
        fixed = _batch_eye(B, self.NUM_JOINTS, 3, J_rel.device, J_rel.dtype)
        mask = torch.zeros(self.NUM_JOINTS, device=J_rel.device, dtype=torch.bool)
        mask[self.fixed_orientation_joints] = True
        Gk = torch.where(mask.view(1, self.NUM_JOINTS, 1, 1), fixed, Gk)

        return Gk @ Gk_learned

    def _compute_skeleton_vertices(
        self,
        B: int,
        J: Float[Tensor, "B 24 3"],
        J_rel: Float[Tensor, "B 24 3"],
        G: Float[Tensor, "B 24 4 4"],
        Rk: Float[Tensor, "B 24 3 3"],
        v_shaped: Float[Tensor, "B V 3"],
    ) -> Float[Tensor, "B Vs 3"]:
        """Compute skeleton mesh vertices."""
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

        Gk = _homog_matrix(Rk, J.unsqueeze(-1)) @ S
        skel_v_h = torch.cat([self._skel_v_template, torch.ones(self._skel_v_template.shape[0], 1, device=device)], -1)
        T = torch.einsum("vj,bjxy->bvxy", self._skel_weights_rigid, Gk)
        skel_aligned = (T @ skel_v_h.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

        # Transform to posed space
        G_inv = _homog_matrix(_batch_eye(B, Nj, 3, device, dtype), -J.unsqueeze(-1))
        G_posed = G @ G_inv
        T = torch.einsum("vj,bjxy->bvxy", self._skel_weights, G_posed)
        return (T @ skel_aligned.unsqueeze(-1)).squeeze(-1)[:, :, :3]


# =============================================================================
# Joint types
# =============================================================================


class _MultiAxisJoint(nn.Module):
    axes: Float[Tensor, "D 3"]

    def __init__(self, axes: list) -> None:
        super().__init__()
        self.register_buffer("axes", torch.tensor(axes, dtype=torch.float32))
        self.n_dof = len(axes)

    def forward(self, q: Float[Tensor, "B D"]) -> Float[Tensor, "B 3 3"]:
        R = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0).expand(q.shape[0], 3, 3)
        for i in range(self.n_dof):
            R = _axisangle_to_matrix(q[:, i : i + 1] * self.axes[i]) @ R
        return R


class _KneeJoint(nn.Module):
    n_dof = 1

    def forward(self, q: Float[Tensor, "B 1"]) -> Float[Tensor, "B 3 3"]:
        axis_angle = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
        axis_angle[:, 2] = -q[:, 0]
        return _axisangle_to_matrix(axis_angle)


class _PinJoint(nn.Module):
    n_dof = 1
    axis: Float[Tensor, "3"]

    def __init__(self, euler_xyz: list) -> None:
        super().__init__()
        euler = torch.tensor(euler_xyz, dtype=torch.float32)
        R = SO3.to_matrix(SO3.from_euler(euler, convention="XYZ"))
        self.register_buffer("axis", R @ torch.tensor([0.0, 0.0, 1.0]))

    def forward(self, q: Float[Tensor, "B 1"]) -> Float[Tensor, "B 3 3"]:
        return _axisangle_to_matrix(q * self.axis)


class _ScapulaJoint(_MultiAxisJoint):
    def __init__(self, axes: list, is_left: bool) -> None:
        super().__init__(axes)
        self.is_left = is_left

    def offset(
        self,
        abd: Float[Tensor, "B"],
        elev: Float[Tensor, "B"],
        thorax_w: Float[Tensor, "B"],
        thorax_h: Float[Tensor, "B"],
    ) -> Float[Tensor, "B 3 1"]:
        def pos(a, e, flip):
            if flip:
                a, e = -a, -e
            rx = thorax_w / 4 * torch.cos(e - np.pi / 4)
            sign = 1 if flip else -1
            return torch.stack(
                [
                    sign * rx * torch.cos(a),
                    -thorax_h / 2 * torch.sin(e - np.pi / 4),
                    thorax_w / 4 * torch.sin(a),
                ],
                1,
            )

        zero = torch.zeros_like(abd)
        return (pos(abd, elev, self.is_left) - pos(zero, zero, self.is_left)).unsqueeze(-1)


class _SpineJoint(_MultiAxisJoint):
    def offset(
        self, yaw: Float[Tensor, "B"], pitch: Float[Tensor, "B"], height: Float[Tensor, "B"]
    ) -> Float[Tensor, "B 3 1"]:
        def arc(angle, t, length):
            theta = angle * t
            y = length * t * torch.sinc(theta / np.pi)
            x = 0.5 * length * angle * t**2 * torch.sinc(theta / (2 * np.pi)) ** 2
            return x, y

        t = torch.ones_like(yaw)
        x1, y1 = arc(yaw, t, height)
        x2, y2 = arc(pitch, t, height)

        zero = torch.zeros_like(yaw)
        x1_0, y1_0 = arc(zero, t, height)
        x2_0, y2_0 = arc(zero, t, height)

        dx = torch.stack([-x1 + x1_0, y1 - y1_0 + y2 - y2_0, -x2 + x2_0], 1)
        return dx.unsqueeze(-1)


# =============================================================================
# Utilities
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
    """Build [B, J, 4, 4] homogeneous matrix from rotation [B, J, 3, 3] and translation [B, J, 3, 1]."""
    pad = R.new_tensor([0, 0, 0, 1]).expand(*R.shape[:2], 1, 4)
    return torch.cat((torch.cat((R, t), -1), pad), 2)


def _batch_eye(B: int, N: int, d: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "B N d d"]:
    return torch.eye(d, device=device, dtype=dtype).view(1, 1, d, d).expand(B, N, d, d)


def _skew(v: Float[Tensor, "B N 3"]) -> Float[Tensor, "B N 3 3"]:
    """Skew-symmetric matrix from vector: [B, N, 3] -> [B, N, 3, 3]."""
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
    """Rotation matrix that rotates normalized vectors a to b. Shape [B, N, 3] -> [B, N, 3, 3]."""
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

# SKEL uses SMPL-compatible pose blend shapes - this maps SKEL joints to SMPL joints
_SMPL_JOINT_MAP = [0, 2, 5, 8, 8, 11, 1, 4, 7, 7, 10, 3, 6, 15, 14, 17, 19, 0, 21, 13, 16, 18, 0, 20]

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
    pelvis_rotation: Float[Tensor, "B 3"] | None = None,
    pelvis_translation: Float[Tensor, "B 3"] | None = None,
) -> dict[str, Tensor | None]:
    """Convert native SKEL args to forward_* kwargs.

    Native format uses body_pose + pelvis_rotation/translation separately.
    API format uses pose (with pelvis rotation in first 3 elements) + global_translation.
    """
    pose = body_pose
    if pelvis_rotation is not None:
        pose = torch.cat([pelvis_rotation, body_pose[:, 3:]], dim=1)

    return {
        "shape": shape,
        "pose": pose,
        "global_translation": pelvis_translation,
    }


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    skeleton_mesh: Float[Tensor, "B Vs 3"],
    feet_offset: Float[Tensor, "3"],
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native SKEL format.

    Native format returns joint positions (not transforms) without feet offset.
    """
    return {
        "vertices": vertices - feet_offset,
        "joints": transforms[..., :3, 3] - feet_offset,
        "skeleton_vertices": skeleton_mesh - feet_offset,
    }


def _simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplify mesh using quadric decimation.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] face indices
        target_faces: target number of faces

    Returns:
        new_vertices: [V', 3] simplified vertex positions
        new_faces: [F', 3] simplified face indices
        vertex_map: [V'] index of nearest original vertex for each new vertex
    """
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    # Find nearest original vertex for each new vertex (for attribute mapping)
    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map
