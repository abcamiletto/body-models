from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from .io import get_model_path


class FLAME(BodyModel):
    """FLAME head model with shape, expression, and pose parameters.

    Args:
        model_path: Path to the FLAME model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
        ground_plane: If True (default), dynamically offset mesh so chin is at Y=0
            regardless of head shape. If False, use native FLAME coordinates.

    Forward API:
        forward_vertices(shape, expression, head_pose, root_rotation, root_translation)
        forward_skeleton(shape, expression, head_pose, root_rotation, root_translation)

        shape: [B, 300] shape betas (can use fewer)
        expression: [B, 100] expression coefficients (can use fewer)
        head_pose: [B, 4, 3] axis-angle for neck, jaw, left_eye, right_eye
    """

    NUM_HEAD_JOINTS = 4  # neck, jaw, left_eye, right_eye
    NUM_JOINTS = 5  # root + 4 head joints

    # Buffer type annotations
    v_template: Float[Tensor, "V 3"]
    v_template_full: Float[Tensor, "V_full 3"]
    J_regressor: Float[Tensor, "5 V_full"]
    lbs_weights: Float[Tensor, "V 5"]
    parents: Int[Tensor, "5"]
    _faces: Int[Tensor, "F 3"]
    _kinematic_fronts: list[tuple[list[int], list[int]]]

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        ground_plane: bool = True,
    ):
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path)
        data = _load_model_data(resolved_path)

        # Load full-resolution data
        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)  # (V, 3, 400)
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

        # FLAME 2023 has combined shape (300) + expression (100) in shapedirs
        shapedirs_full_t = torch.as_tensor(shapedirs_full, dtype=torch.float32)
        self.shapedirs_full = nn.Parameter(shapedirs_full_t[:, :, :300], requires_grad=False)
        self.exprdirs_full = nn.Parameter(shapedirs_full_t[:, :, 300:], requires_grad=False)

        # Simplified buffers for mesh output
        self.register_buffer("v_template", torch.as_tensor(v_template))
        self.register_buffer("lbs_weights", torch.as_tensor(lbs_weights))
        self.register_buffer("_faces", torch.as_tensor(faces))

        parents = torch.as_tensor(data["kintree_table"][0], dtype=torch.int32)
        # Fix parent of root (may be -1 or large value in file)
        parents[0] = 0
        self.register_buffer("parents", parents)

        # Precompute kinematic fronts for batched FK
        self._kinematic_fronts = _compute_kinematic_fronts(parents)

        # Blend shapes (simplified resolution)
        shapedirs_t = torch.as_tensor(shapedirs, dtype=torch.float32)
        self.shapedirs = nn.Parameter(shapedirs_t[:, :, :300], requires_grad=False)
        self.exprdirs = nn.Parameter(shapedirs_t[:, :, 300:], requires_grad=False)

        posedirs_t = torch.as_tensor(posedirs, dtype=torch.float32)
        self.posedirs = nn.Parameter(posedirs_t.reshape(-1, posedirs_t.shape[-1]).T, requires_grad=False)

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
    def skin_weights(self) -> Float[Tensor, "V 5"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        if self.ground_plane:
            min_y = self.v_template_full[:, 1].min()
            offset = torch.zeros(3, device=self.v_template.device, dtype=self.v_template.dtype)
            offset[1] = -min_y
            return self.v_template + offset
        return self.v_template

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 N_shape"],
        expression: Float[Tensor, "B N_expr"] | None = None,
        head_pose: Float[Tensor, "B 4 3"] | None = None,
        root_rotation: Float[Tensor, "B 3"] | None = None,
        root_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        B = shape.shape[0] if shape.dim() > 1 and shape.shape[0] > 1 else (
            head_pose.shape[0] if head_pose is not None else 1
        )
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if expression is None:
            expression = torch.zeros((B, 100), device=device, dtype=dtype)
        if head_pose is None:
            head_pose = torch.zeros((B, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype)

        v_t, j_t, pose_matrices, T_world, ground_offset = self._forward_core(
            shape, expression, head_pose.reshape(B, -1), root_rotation
        )
        assert v_t is not None

        # Pose blend shapes
        pose_delta = (pose_matrices[:, 1:] - torch.eye(3, device=device, dtype=dtype)).reshape(B, -1)
        v_shaped = v_t + (pose_delta @ self.posedirs).reshape(B, -1, 3)

        # Linear blend skinning
        R_world = T_world[..., :3, :3]
        t_world = T_world[..., :3, 3]
        W_R = torch.einsum("vj,bjkl->bvkl", self.lbs_weights, R_world)
        W_t = torch.einsum("vj,bjk->bvk", self.lbs_weights, t_world - (R_world @ j_t[..., None]).squeeze(-1))
        v_posed = (W_R @ v_shaped[..., None]).squeeze(-1) + W_t

        # Apply root translation
        if root_translation is not None:
            v_posed = v_posed + root_translation[:, None]

        # Apply global transform (post-transform around origin)
        v_posed = self._apply_global_transform(v_posed, global_rotation, global_translation)

        # Apply dynamic ground offset (per-batch element)
        return v_posed + ground_offset[:, None]

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 N_shape"],
        expression: Float[Tensor, "B N_expr"] | None = None,
        head_pose: Float[Tensor, "B 4 3"] | None = None,
        root_rotation: Float[Tensor, "B 3"] | None = None,
        root_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 5 4 4"]:
        """Compute skeleton joint transforms [B, 5, 4, 4]."""
        B = shape.shape[0] if shape.dim() > 1 and shape.shape[0] > 1 else (
            head_pose.shape[0] if head_pose is not None else 1
        )
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if expression is None:
            expression = torch.zeros((B, 100), device=device, dtype=dtype)
        if head_pose is None:
            head_pose = torch.zeros((B, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype)

        _, _, _, T_world, ground_offset = self._forward_core(
            shape, expression, head_pose.reshape(B, -1), root_rotation, skeleton_only=True
        )

        # Apply root translation
        if root_translation is not None:
            T_world = T_world.clone()
            T_world[..., :3, 3] = T_world[..., :3, 3] + root_translation[:, None]

        # Apply global transform (post-transform around origin)
        T_world = self._apply_global_transform_to_skeleton(T_world, global_rotation, global_translation)

        # Apply dynamic ground offset (per-batch element)
        T_world[..., :3, 3] = T_world[..., :3, 3] + ground_offset[:, None]
        return T_world

    def _forward_core(
        self,
        shape: Float[Tensor, "B N_shape"],
        expression: Float[Tensor, "B N_expr"],
        head_pose: Float[Tensor, "B 12"],  # 4 joints * 3
        root_rotation: Float[Tensor, "B 3"] | None = None,
        skeleton_only: bool = False,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        """Core forward pass: returns (v_t, j_t, pose_matrices, T_world, ground_offset)."""
        B = head_pose.shape[0]
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)

        # Build full pose with root rotation
        if root_rotation is None:
            root = torch.zeros((B, 3), device=device, dtype=dtype)
        else:
            root = root_rotation
        pose = torch.cat([root, head_pose], dim=-1).reshape(B, -1, 3)
        pose_matrices = SO3.to_matrix(SO3.from_axis_angle(pose))

        # Joint locations (use full-resolution mesh for accurate skeleton)
        shape_dim = min(shape.shape[-1], self.shapedirs_full.shape[-1])
        expr_dim = min(expression.shape[-1], self.exprdirs_full.shape[-1])
        v_t_full = self.v_template_full.unsqueeze(0).expand(B, -1, -1).clone()
        v_t_full = v_t_full + torch.einsum("bi,vdi->bvd", shape[:, :shape_dim], self.shapedirs_full[:, :, :shape_dim])
        v_t_full = v_t_full + torch.einsum("bi,vdi->bvd", expression[:, :expr_dim], self.exprdirs_full[:, :, :expr_dim])
        j_t = torch.einsum("bvd,jv->bjd", v_t_full, self.J_regressor)

        # Compute dynamic ground offset from shaped vertices (rest pose)
        if self.ground_plane:
            min_y = v_t_full[..., 1].min(dim=-1).values  # [B]
            ground_offset = torch.zeros((B, 3), device=device, dtype=dtype)
            ground_offset[:, 1] = -min_y
        else:
            ground_offset = torch.zeros((B, 3), device=device, dtype=dtype)

        # Shape blend shapes (simplified mesh for output) - skip if skeleton_only
        if skeleton_only:
            v_t = None
        else:
            v_t = self.v_template.unsqueeze(0).expand(B, -1, -1).clone()
            v_t = v_t + torch.einsum("bi,vdi->bvd", shape[:, :shape_dim], self.shapedirs[:, :, :shape_dim])
            v_t = v_t + torch.einsum("bi,vdi->bvd", expression[:, :expr_dim], self.exprdirs[:, :, :expr_dim])

        # Forward kinematics (batched using kinematic fronts)
        t_local = torch.zeros((B, self.NUM_JOINTS, 3), device=device, dtype=dtype)
        t_local[:, 0] = j_t[:, 0]
        t_local[:, 1:] = j_t[:, 1:] - j_t[:, self.parents[1:]]

        T_world = _batched_forward_kinematics(pose_matrices, t_local, self._kinematic_fronts)

        return v_t, j_t, pose_matrices, T_world, ground_offset

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
            "shape": torch.zeros((1, 300), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, 100), device=device, dtype=dtype),
            "head_pose": torch.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), device=device, dtype=dtype),
            "root_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "root_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }


def from_native_args(
    shape: Float[Tensor, "B N_shape"],
    expression: Float[Tensor, "B N_expr"],
    head_pose: Float[Tensor, "B 12"],
    root_rotation: Float[Tensor, "B 3"] | None = None,
    root_translation: Float[Tensor, "B 3"] | None = None,
) -> dict[str, Tensor | None]:
    """Convert native FLAME args to forward_* kwargs.

    Native format uses flat head_pose tensor.
    API format uses reshaped head_pose [B, 4, 3].
    """
    return {
        "shape": shape,
        "expression": expression,
        "head_pose": head_pose.reshape(-1, 4, 3),
        "root_rotation": root_rotation,
        "root_translation": root_translation,
    }


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native FLAME format.

    Native format returns joint positions instead of transforms.
    Use ground_plane=False in the FLAME constructor if you need outputs
    compatible with the official smplx library.

    Args:
        vertices: [B, V, 3] mesh vertices.
        transforms: [B, J, 4, 4] joint transforms.
    """
    return {
        "vertices": vertices,
        "joints": transforms[..., :3, 3],
    }


def _load_model_data(model_path: Path) -> dict:
    """Load FLAME model data from a .pkl or .npz file."""
    if model_path.suffix == ".npz":
        return dict(np.load(model_path, allow_pickle=True))

    import pickle

    with open(model_path, "rb") as f:
        model_data = pickle.load(f, encoding="latin1")

    # Handle scipy sparse matrices
    if hasattr(model_data.get("J_regressor"), "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


def _compute_kinematic_fronts(parents: Int[Tensor, "J"]) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK."""
    n_joints = len(parents)
    depths = [-1] * n_joints
    depths[0] = 0  # Root

    for i in range(1, n_joints):
        d = 0
        j = i
        while j != 0:
            j = parents[j].item()
            d += 1
        depths[i] = d

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
    """Batched forward kinematics using precomputed kinematic fronts."""
    B, J = R.shape[:2]
    device, dtype = R.device, R.dtype

    R_world = [None] * J
    t_world = [None] * J
    R_world[0] = R[:, 0]
    t_world[0] = t[:, 0]

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
    """Simplify mesh using quadric decimation."""
    import pyfqmr
    from scipy.spatial import KDTree

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(vertices, faces)
    simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True)
    new_vertices, new_faces, _ = simplifier.getMesh()

    new_vertices = np.asarray(new_vertices, dtype=np.float32)
    new_faces = np.asarray(new_faces, dtype=np.int32)

    tree = KDTree(vertices)
    _, vertex_map = tree.query(new_vertices)
    vertex_map = np.asarray(vertex_map, dtype=np.int64)

    return new_vertices, new_faces, vertex_map
