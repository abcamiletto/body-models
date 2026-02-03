from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from .io import get_model_path



class SMPLX(BodyModel):
    """SMPL-X body model with expressive hands and face.

    Args:
        model_path: Path to the SMPL-X model file or directory.
        gender: One of "neutral", "male", or "female".
        flat_hand_mean: If True, mean hand pose is zero (flat hands).
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
        ground_plane: If True (default), dynamically offset mesh so feet are at Y=0
            regardless of body shape. If False, use native SMPL-X coordinates.

    Forward API:
        forward_vertices(shape, body_pose, hand_pose, head_pose, expression, ...)
        forward_skeleton(shape, body_pose, hand_pose, head_pose, expression, ...)

        shape: [B, 10] body shape betas
        body_pose: [B, 21, 3] axis-angle per body joint
        hand_pose: [B, 30, 3] axis-angle per hand joint (left 15 + right 15)
        head_pose: [B, 3, 3] axis-angle for jaw + left eye + right eye
        expression: [B, 10] facial expression coefficients
    """

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30  # 15 per hand
    NUM_HEAD_JOINTS = 3  # jaw, left eye, right eye
    NUM_JOINTS = 55  # 22 body + 30 hands + 3 head

    # Buffer type annotations
    v_template: Float[Tensor, "V 3"]
    v_template_full: Float[Tensor, "V_full 3"]
    J_regressor: Float[Tensor, "55 V_full"]
    lbs_weights: Float[Tensor, "V 55"]
    hand_mean: Float[Tensor, "2 45"]
    parents: Int[Tensor, "55"]
    _faces: Int[Tensor, "F 3"]
    _shape_expr_dirs_flat_T: Float[Tensor, "20 V*3"]
    _shape_expr_dirs_full_flat_T: Float[Tensor, "20 V_full*3"]
    _kinematic_fronts: list[tuple[list[int], list[int]]]

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: str = "neutral",
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        ground_plane: bool = True,
        use_hand_pca: bool = False,  # Accepted for compatibility, not used
    ):
        assert gender in ("neutral", "male", "female")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()
        self.gender = gender
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path, gender)
        data = np.load(resolved_path, allow_pickle=True)

        # Load full-resolution data
        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = _simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full
            shapedirs = shapedirs_full

        # Full-resolution buffers for skeleton computation
        self.register_buffer("v_template_full", torch.as_tensor(v_template_full))
        self.register_buffer("J_regressor", torch.as_tensor(J_regressor))
        shapedirs_full_t = torch.as_tensor(shapedirs_full, dtype=torch.float32)
        self.shapedirs_full = nn.Parameter(shapedirs_full_t[:, :, :300], requires_grad=False)
        self.exprdirs_full = nn.Parameter(shapedirs_full_t[:, :, 300:400], requires_grad=False)

        # Simplified buffers for mesh output
        self.register_buffer("v_template", torch.as_tensor(v_template))
        self.register_buffer("lbs_weights", torch.as_tensor(lbs_weights))
        self.register_buffer("_faces", torch.as_tensor(faces))

        parents = torch.as_tensor(data["kintree_table"][0], dtype=torch.int32)
        self.register_buffer("parents", parents)

        # Precompute kinematic fronts for batched FK
        self._kinematic_fronts = _compute_kinematic_fronts(parents)

        # Blend shapes (simplified resolution)
        shapedirs_t = torch.as_tensor(shapedirs, dtype=torch.float32)
        self.shapedirs = nn.Parameter(shapedirs_t[:, :, :300], requires_grad=False)
        self.exprdirs = nn.Parameter(shapedirs_t[:, :, 300:400], requires_grad=False)

        posedirs_t = torch.as_tensor(posedirs, dtype=torch.float32)
        self.posedirs = nn.Parameter(posedirs_t.reshape(-1, posedirs_t.shape[-1]).T, requires_grad=False)

        # Precompute flattened shape+expr dirs for fast path
        self._shape_expr_dims = (10, 10)
        shape_dim, expr_dim = self._shape_expr_dims
        shape_expr_dirs = torch.cat([self.shapedirs[:, :, :shape_dim], self.exprdirs[:, :, :expr_dim]], dim=-1)
        self.register_buffer(
            "_shape_expr_dirs_flat_T", shape_expr_dirs.reshape(-1, shape_dim + expr_dim).T.contiguous()
        )
        # Full-resolution version for skeleton
        shape_expr_dirs_full = torch.cat(
            [self.shapedirs_full[:, :, :shape_dim], self.exprdirs_full[:, :, :expr_dim]], dim=-1
        )
        self.register_buffer(
            "_shape_expr_dirs_full_flat_T", shape_expr_dirs_full.reshape(-1, shape_dim + expr_dim).T.contiguous()
        )

        # Hand pose mean
        hand_mean = torch.stack(
            [
                torch.as_tensor(data["hands_meanl"], dtype=torch.float32),
                torch.as_tensor(data["hands_meanr"], dtype=torch.float32),
            ]
        )
        if flat_hand_mean:
            hand_mean.zero_()
        self.register_buffer("hand_mean", hand_mean)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Tensor, "V 55"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        if self.ground_plane:
            # Compute feet offset for identity shape
            min_y = self.v_template_full[:, 1].min()
            offset = torch.zeros(3, device=self.v_template.device, dtype=self.v_template.dtype)
            offset[1] = -min_y
            return self.v_template + offset
        return self.v_template

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 21 3"],
        hand_pose: Float[Tensor, "B 30 3"],
        head_pose: Float[Tensor, "B 3 3"],
        expression: Float[Tensor, "B 10"] | None = None,
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        pelvis_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        B = body_pose.shape[0]
        device, dtype = body_pose.device, body_pose.dtype

        if expression is None:
            expression = torch.zeros((B, 10), device=device, dtype=dtype)

        v_t, j_t, pose_matrices, T_world, feet_offset = self._forward_core(
            shape,
            expression,
            body_pose.reshape(B, -1),
            hand_pose.reshape(B, -1),
            head_pose.reshape(B, -1),
            pelvis_rotation,
        )
        assert v_t is not None

        # Pose blend shapes
        pose_delta = (pose_matrices[:, 1:] - torch.eye(3, device=device, dtype=dtype)).reshape(B, -1)
        v_shaped = v_t + (pose_delta @ self.posedirs).reshape(B, -1, 3)

        # Linear blend skinning (optimized: compute weighted transforms, not per-joint offsets)
        R_world = T_world[..., :3, :3]
        t_world = T_world[..., :3, 3]
        W_R = torch.einsum("vj,bjkl->bvkl", self.lbs_weights, R_world)
        W_t = torch.einsum("vj,bjk->bvk", self.lbs_weights, t_world - (R_world @ j_t[..., None]).squeeze(-1))
        v_posed = (W_R @ v_shaped[..., None]).squeeze(-1) + W_t

        # Apply pelvis translation
        if pelvis_translation is not None:
            v_posed = v_posed + pelvis_translation[:, None]

        # Apply global transform (post-transform around origin)
        v_posed = self._apply_global_transform(v_posed, global_rotation, global_translation)

        # Apply dynamic feet offset (per-batch element)
        return v_posed + feet_offset[:, None]

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 21 3"],
        hand_pose: Float[Tensor, "B 30 3"],
        head_pose: Float[Tensor, "B 3 3"],
        expression: Float[Tensor, "B 10"] | None = None,
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        pelvis_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 55 4 4"]:
        """Compute skeleton joint transforms [B, 55, 4, 4]."""
        B = body_pose.shape[0]
        device, dtype = body_pose.device, body_pose.dtype

        if expression is None:
            expression = torch.zeros((B, 10), device=device, dtype=dtype)

        _, _, _, T_world, feet_offset = self._forward_core(
            shape,
            expression,
            body_pose.reshape(B, -1),
            hand_pose.reshape(B, -1),
            head_pose.reshape(B, -1),
            pelvis_rotation,
            skeleton_only=True,
        )

        # Apply pelvis translation
        if pelvis_translation is not None:
            T_world = T_world.clone()
            T_world[..., :3, 3] = T_world[..., :3, 3] + pelvis_translation[:, None]

        # Apply global transform (post-transform around origin)
        T_world = self._apply_global_transform_to_skeleton(T_world, global_rotation, global_translation)

        # Apply dynamic feet offset (per-batch element)
        T_world[..., :3, 3] = T_world[..., :3, 3] + feet_offset[:, None]
        return T_world

    def _forward_core(
        self,
        shape: Float[Tensor, "B 10"],
        expression: Float[Tensor, "B 10"],
        body_pose: Float[Tensor, "B 63"],
        hand_pose: Float[Tensor, "B 90"],
        head_pose: Float[Tensor, "B 9"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        skeleton_only: bool = False,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        """Core forward pass: returns (v_t, j_t, pose_matrices, T_world, feet_offset).

        Args:
            skeleton_only: If True, skip simplified mesh computation (v_t=None).
        """
        B = body_pose.shape[0]
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)

        # Apply hand pose mean
        lh, rh = hand_pose.chunk(2, dim=-1)
        hand_pose = torch.cat([lh + self.hand_mean[0], rh + self.hand_mean[1]], dim=-1)

        # Build full pose with pelvis/root rotation
        if pelvis_rotation is None:
            pelvis = torch.zeros((B, 3), device=device, dtype=dtype)
        else:
            pelvis = pelvis_rotation
        pose = torch.cat([pelvis, body_pose, head_pose, hand_pose], dim=-1).reshape(B, -1, 3)
        pose_matrices = SO3.to_matrix(SO3.from_axis_angle(pose))

        # Joint locations (use full-resolution mesh for accurate skeleton)
        shape_dim = shape.shape[-1]
        expr_dim = expression.shape[-1]
        shape_comps = torch.cat([shape, expression], dim=-1)
        if (shape_dim, expr_dim) == self._shape_expr_dims:
            v_t_full = self.v_template_full + (shape_comps @ self._shape_expr_dirs_full_flat_T).reshape(B, -1, 3)
        else:
            sd_full = torch.cat([self.shapedirs_full[:, :, :shape_dim], self.exprdirs_full[:, :, :expr_dim]], dim=-1)
            v_t_full = self.v_template_full + torch.einsum("bi,vdi->bvd", shape_comps, sd_full)
        j_t = torch.einsum("bvd,jv->bjd", v_t_full, self.J_regressor)

        # Compute dynamic feet offset from shaped vertices (rest pose)
        # This ensures feet are at Y=0 regardless of body shape
        if self.ground_plane:
            min_y = v_t_full[..., 1].min(dim=-1).values  # [B]
            feet_offset = torch.zeros((B, 3), device=device, dtype=dtype)
            feet_offset[:, 1] = -min_y
        else:
            feet_offset = torch.zeros((B, 3), device=device, dtype=dtype)

        # Shape blend shapes (simplified mesh for output) - skip if skeleton_only
        if skeleton_only:
            v_t = None
        elif (shape_dim, expr_dim) == self._shape_expr_dims:
            v_t = self.v_template + (shape_comps @ self._shape_expr_dirs_flat_T).reshape(B, -1, 3)
        else:
            sd = torch.cat([self.shapedirs[:, :, :shape_dim], self.exprdirs[:, :, :expr_dim]], dim=-1)
            v_t = self.v_template + torch.einsum("bi,vdi->bvd", shape_comps, sd)

        # Forward kinematics (batched using kinematic fronts)
        t_local = torch.zeros((B, self.NUM_JOINTS, 3), device=device, dtype=dtype)
        t_local[:, 0] = j_t[:, 0]
        t_local[:, 1:] = j_t[:, 1:] - j_t[:, self.parents[1:]]

        T_world = _batched_forward_kinematics(pose_matrices, t_local, self._kinematic_fronts)

        return v_t, j_t, pose_matrices, T_world, feet_offset

    def _apply_global_transform(
        self,
        points: Float[Tensor, "B N 3"],
        rotation: Float[Tensor, "B 3"] | None,
        translation: Float[Tensor, "B 3"] | None,
    ) -> Float[Tensor, "B N 3"]:
        """Apply global rotation and translation to points [B, N, 3]."""
        if rotation is not None:
            R = SO3.to_matrix(SO3.from_axis_angle(rotation))
            points = (R @ points.mT).mT
        if translation is not None:
            points = points + translation[:, None]
        return points

    def _apply_global_transform_to_skeleton(
        self,
        T: Float[Tensor, "B J 4 4"],
        rotation: Float[Tensor, "B 3"] | None,
        translation: Float[Tensor, "B 3"] | None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Apply global rotation and translation to skeleton transforms [B, J, 4, 4]."""
        if rotation is None and translation is None:
            return T
        T = T.clone()
        if rotation is not None:
            R = SO3.to_matrix(SO3.from_axis_angle(rotation))
            T[..., :3, 3] = (R @ T[..., :3, 3].mT).mT
            T[..., :3, :3] = R[:, None] @ T[..., :3, :3]
        if translation is not None:
            T[..., :3, 3] = T[..., :3, 3] + translation[:, None]
        return T

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        """Get rest pose parameters."""
        device = self.v_template.device
        return {
            "shape": torch.zeros((1, 10), device=device, dtype=dtype),
            "body_pose": torch.zeros((batch_size, self.NUM_BODY_JOINTS, 3), device=device, dtype=dtype),
            "hand_pose": torch.zeros((batch_size, self.NUM_HAND_JOINTS, 3), device=device, dtype=dtype),
            "head_pose": torch.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, 10), device=device, dtype=dtype),
            "pelvis_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "pelvis_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }


def from_native_args(
    shape: Float[Tensor, "B 10"],
    expression: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 63"],
    hand_pose: Float[Tensor, "B 90"],
    head_pose: Float[Tensor, "B 9"],
    pelvis_rotation: Float[Tensor, "B 3"] | None = None,
    pelvis_translation: Float[Tensor, "B 3"] | None = None,
) -> dict[str, Tensor | None]:
    """Convert native SMPLX args to forward_* kwargs.

    Native format uses flat pose tensors.
    API format uses reshaped pose tensors [B, J, 3].
    """
    return {
        "shape": shape,
        "body_pose": body_pose.reshape(-1, 21, 3),
        "hand_pose": hand_pose.reshape(-1, 30, 3),
        "head_pose": head_pose.reshape(-1, 3, 3),
        "expression": expression,
        "pelvis_rotation": pelvis_rotation,
        "pelvis_translation": pelvis_translation,
    }


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native SMPLX format.

    Native format returns joint positions instead of transforms.
    Use ground_plane=False in the SMPLX constructor if you need outputs
    compatible with the official smplx library.

    Args:
        vertices: [B, V, 3] mesh vertices.
        transforms: [B, J, 4, 4] joint transforms.
    """
    return {
        "vertices": vertices,
        "joints": transforms[..., :3, 3],
    }


def _compute_kinematic_fronts(parents: Int[Tensor, "J"]) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK.

    Returns list of (joint_indices, parent_indices) tuples, one per depth level.
    Joints at the same depth can be processed in parallel.
    """
    n_joints = len(parents)
    depths = [-1] * n_joints
    depths[0] = 0  # Root

    # Compute depth of each joint
    for i in range(1, n_joints):
        d = 0
        j = i
        while j != 0:
            j = parents[j].item()
            d += 1
        depths[i] = d

    # Group joints by depth
    max_depth = max(depths)
    fronts = []
    for d in range(1, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        parent_indices = [parents[j].item() for j in joints]
        fronts.append((joints, parent_indices))

    return fronts


def _batched_forward_kinematics(
    R: Float[Tensor, "B J 3 3"],
    t: Float[Tensor, "B J 3"],
    fronts: list[tuple[list[int], list[int]]],
) -> Float[Tensor, "B J 4 4"]:
    """Batched forward kinematics using precomputed kinematic fronts.

    Args:
        R: [B, J, 3, 3] local rotation matrices
        t: [B, J, 3] local translations
        fronts: precomputed kinematic fronts from _compute_kinematic_fronts

    Returns:
        T_world: [B, J, 4, 4] world transforms
    """
    B, J = R.shape[:2]
    device, dtype = R.device, R.dtype

    R_world = [None] * J
    t_world = [None] * J
    R_world[0] = R[:, 0]
    t_world[0] = t[:, 0]

    # Process each depth level in parallel
    for joints, parent_indices in fronts:
        R_parent = torch.stack([R_world[i] for i in parent_indices], dim=1)
        t_parent = torch.stack([t_world[i] for i in parent_indices], dim=1)
        R_local = R[:, joints]
        t_local = t[:, joints]

        R_cur = R_parent @ R_local
        t_cur = t_parent + (R_parent @ t_local[..., None]).squeeze(-1)
        for idx, joint in enumerate(joints):
            R_world[joint] = R_cur[:, idx]
            t_world[joint] = t_cur[:, idx]

    R_world = torch.stack(R_world, dim=1)
    t_world = torch.stack(t_world, dim=1)
    T_world = torch.zeros((B, J, 4, 4), device=device, dtype=dtype)
    T_world[..., 3, 3] = 1.0
    T_world[..., :3, :3] = R_world
    T_world[..., :3, 3] = t_world
    return T_world


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
