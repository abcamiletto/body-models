from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from .io import get_model_path

# Feet offset (Y) for floor alignment, per gender.
# Currently using neutral values for all genders.
_FEET_OFFSET_Y = {
    "neutral": 1.1618428230285645,
    "male": 1.1618428230285645,
    "female": 1.1618428230285645,
}


class SMPL(BodyModel):
    """SMPL body model with shape and pose parameters.

    Args:
        model_path: Path to the SMPL model file or directory.
        gender: One of "neutral", "male", or "female".
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.

    Forward API:
        forward_vertices(shape, body_pose, global_rotation, global_translation)
        forward_skeleton(shape, body_pose, global_rotation, global_translation)

        shape: [B, 10] body shape betas
        body_pose: [B, 23, 3] axis-angle per body joint (excluding pelvis)
    """

    NUM_BODY_JOINTS = 23  # excluding pelvis
    NUM_JOINTS = 24  # pelvis + 23 body joints

    # Buffer type annotations
    v_template: Float[Tensor, "V 3"]
    v_template_full: Float[Tensor, "V_full 3"]
    J_regressor: Float[Tensor, "24 V_full"]
    lbs_weights: Float[Tensor, "V 24"]
    parents: Int[Tensor, "24"]
    _faces: Int[Tensor, "F 3"]
    _feet_offset: Float[Tensor, "3"]
    _kinematic_fronts: list[tuple[list[int], list[int]]]

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: str = "neutral",
        simplify: float = 1.0,
    ):
        assert gender in ("neutral", "male", "female")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()
        self.gender = gender

        resolved_path = get_model_path(model_path, gender)
        data = _load_model_data(resolved_path)

        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
        shapedirs = shapedirs_full
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)

        # Apply mesh simplification if requested
        # Keep full-resolution data for skeleton, simplify only for mesh output
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = _simplify_mesh(v_template_full, faces, target_faces)

            # Map vertex attributes using nearest neighbor from original mesh
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full

        # Full-resolution buffers for skeleton computation (joint positions)
        self.register_buffer("v_template_full", torch.as_tensor(v_template_full))
        self.register_buffer("J_regressor", torch.as_tensor(J_regressor))
        self.shapedirs_full = nn.Parameter(torch.as_tensor(shapedirs_full), requires_grad=False)

        # Simplified buffers for mesh output
        self.register_buffer("v_template", torch.as_tensor(v_template))
        self.register_buffer("lbs_weights", torch.as_tensor(lbs_weights))
        self.register_buffer("_faces", torch.as_tensor(faces))

        parents = torch.as_tensor(data["kintree_table"][0], dtype=torch.int32)
        self.register_buffer("parents", parents)

        # Precompute kinematic fronts for batched FK
        self._kinematic_fronts = _compute_kinematic_fronts(parents)

        # Blend shapes (simplified resolution)
        self.shapedirs = nn.Parameter(torch.as_tensor(shapedirs), requires_grad=False)
        self.posedirs = nn.Parameter(torch.as_tensor(posedirs.reshape(-1, posedirs.shape[-1]).T), requires_grad=False)

        # Feet offset for floor alignment
        y_offset = _FEET_OFFSET_Y[gender]
        self.register_buffer("_feet_offset", torch.tensor([0.0, y_offset, 0.0], dtype=torch.float32))

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
    def skin_weights(self) -> Float[Tensor, "V 24"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.v_template + self._feet_offset

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 23 3"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        pelvis_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        B = body_pose.shape[0]
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        v_t, j_t, pose_matrices, T_world = self._forward_core(shape, body_pose.reshape(B, -1), pelvis_rotation)
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
        return v_posed + self._feet_offset

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 10"],
        body_pose: Float[Tensor, "B 23 3"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        pelvis_translation: Float[Tensor, "B 3"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B 24 4 4"]:
        """Compute skeleton joint transforms [B, 24, 4, 4]."""
        B = body_pose.shape[0]
        _, _, _, T_world = self._forward_core(shape, body_pose.reshape(B, -1), pelvis_rotation, skeleton_only=True)

        # Apply pelvis translation
        if pelvis_translation is not None:
            T_world = T_world.clone()
            T_world[..., :3, 3] = T_world[..., :3, 3] + pelvis_translation[:, None]

        # Apply global transform (post-transform around origin)
        T_world = self._apply_global_transform_to_skeleton(T_world, global_rotation, global_translation)
        T_world[..., :3, 3] = T_world[..., :3, 3] + self._feet_offset
        return T_world

    def _forward_core(
        self,
        shape: Float[Tensor, "B 10"],
        body_pose: Float[Tensor, "B 69"],
        pelvis_rotation: Float[Tensor, "B 3"] | None = None,
        skeleton_only: bool = False,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
        """Core forward pass: returns (v_t, j_t, pose_matrices, T_world).

        Args:
            skeleton_only: If True, skip simplified mesh computation (v_t=None).
        """
        B = body_pose.shape[0]
        device, dtype = self.J_regressor.device, self.J_regressor.dtype

        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)

        # Build full pose with pelvis/root rotation
        if pelvis_rotation is None:
            pelvis = torch.zeros((B, 3), device=device, dtype=dtype)
        else:
            pelvis = pelvis_rotation
        pose = torch.cat([pelvis, body_pose], dim=-1).reshape(B, -1, 3)
        pose_matrices = SO3.to_matrix(SO3.from_axis_angle(pose))

        # Joint locations (use full-resolution mesh for accurate skeleton)
        v_t_full = self.v_template_full + torch.einsum(
            "bi,vdi->bvd", shape, self.shapedirs_full[:, :, : shape.shape[-1]]
        )
        j_t = torch.einsum("bvd,jv->bjd", v_t_full, self.J_regressor)

        # Shape blend shapes (simplified mesh for output) - skip if skeleton_only
        if skeleton_only:
            v_t = None
        else:
            v_t = self.v_template + torch.einsum("bi,vdi->bvd", shape, self.shapedirs[:, :, : shape.shape[-1]])

        # Forward kinematics (batched using kinematic fronts)
        t_local = torch.zeros((B, self.NUM_JOINTS, 3), device=device, dtype=dtype)
        t_local[:, 0] = j_t[:, 0]
        t_local[:, 1:] = j_t[:, 1:] - j_t[:, self.parents[1:]]

        T_world = _batched_forward_kinematics(pose_matrices, t_local, self._kinematic_fronts)

        return v_t, j_t, pose_matrices, T_world

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
            "pelvis_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "pelvis_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }


def from_native_args(
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 69"],
    pelvis_rotation: Float[Tensor, "B 3"] | None = None,
    pelvis_translation: Float[Tensor, "B 3"] | None = None,
) -> dict[str, Tensor | None]:
    """Convert native SMPL args to forward_* kwargs.

    Native format uses flat body_pose tensor.
    API format uses reshaped body_pose [B, 23, 3].
    """
    return {
        "shape": shape,
        "body_pose": body_pose.reshape(-1, 23, 3),
        "pelvis_rotation": pelvis_rotation,
        "pelvis_translation": pelvis_translation,
    }


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    gender: str = "neutral",
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native SMPL format.

    Native format returns joint positions (not transforms) and doesn't include feet offset.
    """
    feet_offset = torch.tensor([0.0, _FEET_OFFSET_Y[gender], 0.0], device=vertices.device, dtype=vertices.dtype)
    return {
        "vertices": vertices - feet_offset,
        "joints": transforms[..., :3, 3] - feet_offset,
    }


def _load_model_data(model_path: Path) -> dict:
    """Load SMPL model data from a .pkl or .npz file."""
    if model_path.suffix == ".npz":
        return dict(np.load(model_path, allow_pickle=True))

    import pickle

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as e:
        if "chumpy" in str(e):
            npz_path = model_path.with_suffix(".npz")
            raise RuntimeError(
                f"This SMPL pkl file requires chumpy to load. "
                f"Convert it to npz format first:\n\n"
                f"  uvx --from body-models convert-smpl-pkl {model_path} {npz_path}\n\n"
                f"Then use the npz file instead."
            ) from None
        raise

    # Handle scipy sparse matrices
    if hasattr(model_data["J_regressor"], "toarray"):
        model_data["J_regressor"] = model_data["J_regressor"].toarray()

    return model_data


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

    # Group joints by depth (include depth 0 for root)
    max_depth = max(depths)
    fronts = []
    for d in range(0, max_depth + 1):
        joints = [i for i in range(n_joints) if depths[i] == d]
        if d == 0:
            parent_indices = [-1] * len(joints)  # Root has no parent
        else:
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

    for joints, parents in fronts:
        if parents[0] < 0:  # Root joints
            for joint in joints:
                R_world[joint] = R[:, joint]
                t_world[joint] = t[:, joint]
            continue

        R_parent = torch.stack([R_world[i] for i in parents], dim=1)
        t_parent = torch.stack([t_world[i] for i in parents], dim=1)
        R_local = R[:, joints]
        t_local = t[:, joints]

        R_cur = R_parent @ R_local
        t_cur = t_parent + (R_parent @ t_local[..., None]).squeeze(-1)
        for idx, joint in enumerate(joints):
            R_world[joint] = R_cur[:, idx]
            t_world[joint] = t_cur[:, idx]

    R_world = torch.stack(R_world, dim=1)
    t_world = torch.stack(t_world, dim=1)
    T = torch.zeros((B, J, 4, 4), device=device, dtype=dtype)
    T[..., :3, :3] = R_world
    T[..., :3, 3] = t_world
    T[..., 3, 3] = 1.0
    return T


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
