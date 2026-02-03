# Derived from: https://github.com/naver/anny
# Original license: Apache 2.0 (https://github.com/naver/anny/blob/main/LICENSE)

import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from ..utils import get_cache_dir
from .io import get_model_path

# Coordinate transform constants (Z-up to Y-up)
_COORD_ROTATION = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_COORD_TRANSLATION = torch.tensor([0.0, 0.852, 0.0])

PHENOTYPE_VARIATIONS = {
    "race": ["african", "asian", "caucasian"],
    "gender": ["male", "female"],
    "age": ["newborn", "baby", "child", "young", "old"],
    "muscle": ["minmuscle", "averagemuscle", "maxmuscle"],
    "weight": ["minweight", "averageweight", "maxweight"],
    "height": ["minheight", "maxheight"],
    "proportions": ["idealproportions", "uncommonproportions"],
    "cupsize": ["mincup", "averagecup", "maxcup"],
    "firmness": ["minfirmness", "averagefirmness", "maxfirmness"],
}
PHENOTYPE_LABELS = [k for k in PHENOTYPE_VARIATIONS if k != "race"] + PHENOTYPE_VARIATIONS["race"]
EXCLUDED_PHENOTYPES = ["cupsize", "firmness"] + PHENOTYPE_VARIATIONS["race"]


class ANNY(BodyModel):
    """ANNY body model with phenotype-based morphology.

    Args:
        model_path: Path to ANNY model directory. Auto-downloads if None.
        cache_dir: Cache directory for preprocessed data.
        rig: Skeleton rig type ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo").
        topology: Mesh topology ("default" or "makehuman").
        all_phenotypes: Include race, cupsize, firmness phenotypes.
        extrapolate_phenotypes: Allow phenotype values outside [0, 1].
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
            Note: Simplified meshes use triangular faces instead of quads.

    Forward API:
        forward_vertices(gender, age, muscle, weight, height, proportions, pose, ...)
        forward_skeleton(gender, age, muscle, weight, height, proportions, pose, ...)

        Phenotype parameters: [B] tensors in [0, 1]
        pose: [B, J, 3] axis-angle per joint
    """

    # Buffer type annotations
    template_vertices: Float[Tensor, "V 3"]
    blendshapes: Float[Tensor, "S V 3"]
    template_bone_heads: Float[Tensor, "J 3"]
    template_bone_tails: Float[Tensor, "J 3"]
    bone_heads_blendshapes: Float[Tensor, "S J 3"]
    bone_tails_blendshapes: Float[Tensor, "S J 3"]
    bone_rolls_rotmat: Float[Tensor, "J 3 3"]
    vertex_bone_weights: Float[Tensor, "V K"]
    vertex_bone_indices: Int[Tensor, "V K"]
    lbs_weights: Float[Tensor, "V J"]
    phenotype_mask: Float[Tensor, "S P"]
    _y_axis: Float[Tensor, "3"]
    _degenerate_rotation: Float[Tensor, "3 3"]
    _coord_rotation: Float[Tensor, "3 3"]
    _coord_translation: Float[Tensor, "3"]

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        cache_dir: Path | str | None = None,
        rig: str = "default",
        topology: str = "default",
        all_phenotypes: bool = False,
        extrapolate_phenotypes: bool = False,
        simplify: float = 1.0,
    ) -> None:
        assert rig in ("default", "default_no_toes", "cmu_mb", "game_engine", "mixamo")
        assert topology in ("default", "makehuman")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

        resolved_path = get_model_path(model_path)
        cache_dir = Path(cache_dir) if cache_dir else get_cache_dir() / "anny" / "preprocessed"

        data = _load_data(resolved_path, cache_dir, rig=rig, eyes=True, tongue=True)

        # Apply topology edits
        if topology == "default":
            data["faces"], data["face_uvs"] = _edit_mesh_faces(data["faces"], data["face_uvs"])

        # Remove unattached vertices
        used_verts = torch.unique(data["faces"].flatten(), sorted=True)
        remap = torch.full((len(data["template_vertices"]),), -1, dtype=torch.int64)
        remap[used_verts] = torch.arange(len(used_verts))

        data["template_vertices"] = data["template_vertices"][used_verts]
        data["vertex_bone_weights"] = data["vertex_bone_weights"][used_verts]
        data["vertex_bone_indices"] = data["vertex_bone_indices"][used_verts]
        data["blendshapes"] = data["blendshapes"][:, used_verts]
        data["faces"] = remap[data["faces"].flatten()].reshape(data["faces"].shape)

        # Trim zero-weight bone columns
        while (data["vertex_bone_weights"].min(dim=-1).values == 0).all():
            keep = data["vertex_bone_weights"].argmin(dim=-1, keepdim=True)
            mask = torch.arange(data["vertex_bone_weights"].shape[1])[None, :] != keep
            data["vertex_bone_weights"] = data["vertex_bone_weights"][mask].reshape(len(used_verts), -1)
            data["vertex_bone_indices"] = data["vertex_bone_indices"][mask].reshape(len(used_verts), -1)

        # Apply mesh simplification if requested
        if simplify > 1.0:
            orig_dtype = data["template_vertices"].dtype

            # Convert quads to triangles: each quad (a,b,c,d) -> triangles (a,b,c), (a,c,d)
            quads = data["faces"]
            tri_faces = torch.cat([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], dim=0)

            # Simplify mesh
            target_faces = int(len(tri_faces) / simplify)
            vertices_np = data["template_vertices"].numpy()
            faces_np = tri_faces.numpy().astype(int)
            new_vertices, new_faces, vertex_map = _simplify_mesh(vertices_np, faces_np, target_faces)

            # Remap per-vertex attributes (preserve original dtype)
            data["template_vertices"] = torch.as_tensor(new_vertices, dtype=orig_dtype)
            data["blendshapes"] = data["blendshapes"][:, vertex_map]
            data["vertex_bone_weights"] = data["vertex_bone_weights"][vertex_map]
            data["vertex_bone_indices"] = data["vertex_bone_indices"][vertex_map]
            data["faces"] = torch.as_tensor(new_faces, dtype=torch.int64)

        # Register buffers
        dtype = data["template_vertices"].dtype
        for key in [
            "template_vertices",
            "blendshapes",
            "template_bone_heads",
            "template_bone_tails",
            "bone_heads_blendshapes",
            "bone_tails_blendshapes",
            "bone_rolls_rotmat",
            "vertex_bone_weights",
            "vertex_bone_indices",
            "phenotype_mask",
        ]:
            self.register_buffer(key, data[key], persistent=False)

        self.register_buffer("_y_axis", torch.tensor([0.0, 1.0, 0.0], dtype=dtype), persistent=False)
        self.register_buffer(
            "_degenerate_rotation", torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=dtype)), persistent=False
        )
        self.register_buffer(
            "_coord_rotation",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=dtype),
            persistent=False,
        )
        self.register_buffer("_coord_translation", torch.tensor([0.0, 0.852, 0.0], dtype=dtype), persistent=False)

        # Precompute dense LBS weights for faster skinning
        V, J = data["vertex_bone_weights"].shape[0], len(data["bone_labels"])
        lbs_weights = torch.zeros(V, J, dtype=dtype)
        lbs_weights.scatter_(1, data["vertex_bone_indices"], data["vertex_bone_weights"])
        self.register_buffer("lbs_weights", lbs_weights, persistent=False)

        self._faces = data["faces"]
        self.bone_parents = data["bone_parents"]
        self.bone_labels = data["bone_labels"]
        self._kinematic_fronts = _build_kinematic_fronts(data["bone_parents"])
        self.extrapolate_phenotypes = extrapolate_phenotypes
        self.all_phenotypes = all_phenotypes
        self.phenotype_labels = (
            PHENOTYPE_LABELS if all_phenotypes else [x for x in PHENOTYPE_LABELS if x not in EXCLUDED_PHENOTYPES]
        )

        # Phenotype interpolation anchors
        self._anchors = nn.ParameterDict(
            {
                "age": nn.Parameter(torch.linspace(-1 / 3, 1.0, 5, dtype=dtype), requires_grad=False),
                **{
                    k: nn.Parameter(
                        torch.linspace(0.0, 1.0, len(PHENOTYPE_VARIATIONS[k]), dtype=dtype), requires_grad=False
                    )
                    for k in ["gender", "muscle", "weight", "height", "proportions", "cupsize", "firmness"]
                },
            }
        )

    @property
    def faces(self) -> Int[Tensor, "F _"]:
        """Face indices. Shape [F, 4] for quads (original) or [F, 3] for triangles (simplified)."""
        return self._faces

    @property
    def num_joints(self) -> int:
        return len(self.bone_labels)

    @property
    def num_vertices(self) -> int:
        return self.template_vertices.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.template_vertices.dtype

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.template_vertices @ self._coord_rotation.T + self._coord_translation

    def forward_vertices(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        pose: Float[Tensor, "B J 3"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3]."""
        pose_T = self._axis_angle_to_transform(pose)
        coeffs, rest_poses, bone_transforms = self._forward_core(
            gender,
            age,
            muscle,
            weight,
            height,
            proportions,
            pose_T,
        )

        # Vertex blendshapes
        rest_verts = self.template_vertices + torch.einsum("bs,svd->bvd", coeffs, self.blendshapes)

        # Linear blend skinning (optimized: compute weighted transforms via einsum)
        R = bone_transforms[..., :3, :3]  # [B, J, 3, 3]
        t = bone_transforms[..., :3, 3]  # [B, J, 3]
        W_R = torch.einsum("vj,bjkl->bvkl", self.lbs_weights, R)  # [B, V, 3, 3]
        W_t = torch.einsum("vj,bjk->bvk", self.lbs_weights, t)  # [B, V, 3]
        vertices = (W_R @ rest_verts[..., None]).squeeze(-1) + W_t

        # Coordinate transform + global
        vertices = vertices @ self._coord_rotation.T + self._coord_translation
        if global_rotation is not None:
            global_rotation = global_rotation.to(vertices.dtype)
            vertices = vertices @ SO3.to_matrix(SO3.from_axis_angle(global_rotation)).mT
        if global_translation is not None:
            global_translation = global_translation.to(vertices.dtype)
            vertices = vertices + global_translation[:, None]
        return vertices

    def forward_skeleton(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        pose: Float[Tensor, "B J 3"],
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute skeleton transforms [B, J, 4, 4]."""
        pose_T = self._axis_angle_to_transform(pose)
        _, bone_poses, _ = self._forward_core(
            gender,
            age,
            muscle,
            weight,
            height,
            proportions,
            pose_T,
        )

        # Coordinate transform
        coord_T = torch.eye(4, device=bone_poses.device, dtype=bone_poses.dtype)
        coord_T[:3, :3] = self._coord_rotation
        coord_T[:3, 3] = self._coord_translation
        transforms = coord_T @ bone_poses

        # Global transform
        if global_rotation is not None or global_translation is not None:
            B = transforms.shape[0]
            G = torch.eye(4, device=transforms.device, dtype=transforms.dtype).expand(B, 4, 4).clone()
            if global_rotation is not None:
                G[:, :3, :3] = SO3.to_matrix(SO3.from_axis_angle(global_rotation))
            if global_translation is not None:
                G[:, :3, 3] = global_translation
            transforms = G[:, None] @ transforms
        return transforms

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        """Get rest pose parameters."""
        device = self.template_vertices.device
        return {
            **{
                k: torch.full((batch_size,), 0.5, device=device, dtype=dtype)
                for k in ["gender", "age", "muscle", "weight", "height", "proportions"]
            },
            "pose": torch.zeros((batch_size, self.num_joints, 3), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    def _axis_angle_to_transform(self, pose: Float[Tensor, "B J 3"]) -> Float[Tensor, "B J 4 4"]:
        """Convert axis-angle pose to 4x4 transforms."""
        R = SO3.to_matrix(SO3.from_axis_angle(pose))
        T = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
        T[..., :3, :3] = R
        T[..., 3, 3] = 1.0
        return T

    def _forward_core(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        pose_T: Float[Tensor, "B J 4 4"],
    ) -> tuple[Float[Tensor, "B S"], Float[Tensor, "B J 4 4"], Float[Tensor, "B J 4 4"]]:
        """Core forward: returns (blendshape_coeffs, bone_poses, bone_transforms)."""
        dtype = self.dtype
        pose_T = pose_T.to(dtype)

        # Phenotype -> blendshape coefficients
        coeffs = self._phenotype_to_coeffs(
            gender.to(dtype),
            age.to(dtype),
            muscle.to(dtype),
            weight.to(dtype),
            height.to(dtype),
            proportions.to(dtype),
        )

        # Rest bone poses from blendshapes
        heads = self.template_bone_heads + torch.einsum("bs,sjd->bjd", coeffs, self.bone_heads_blendshapes)
        tails = self.template_bone_tails + torch.einsum("bs,sjd->bjd", coeffs, self.bone_tails_blendshapes)
        rest_poses = _bone_poses_from_heads_tails(
            heads, tails, self.bone_rolls_rotmat, self._y_axis, self._degenerate_rotation
        )

        # Root parameterization
        root_rest = rest_poses[:, 0]
        base_T = _invert_transform(root_rest)
        delta_T = pose_T.clone()
        root_rot = torch.zeros_like(root_rest)
        root_rot[:, :3, :3] = root_rest[:, :3, :3]
        root_rot[:, 3, 3] = 1.0
        delta_T[:, 0] = pose_T[:, 0] @ root_rot

        # Forward kinematics
        bone_poses, bone_transforms = _forward_kinematics(self._kinematic_fronts, rest_poses, delta_T, base_T)
        return coeffs, bone_poses, bone_transforms

    def _phenotype_to_coeffs(
        self,
        gender: Float[Tensor, "B"],
        age: Float[Tensor, "B"],
        muscle: Float[Tensor, "B"],
        weight: Float[Tensor, "B"],
        height: Float[Tensor, "B"],
        proportions: Float[Tensor, "B"],
        cupsize: float = 0.5,
        firmness: float = 0.5,
        african: float = 0.5,
        asian: float = 0.5,
        caucasian: float = 0.5,
    ) -> Float[Tensor, "B S"]:
        """Convert phenotype parameters to blendshape coefficients."""
        device, dtype = self.phenotype_mask.device, self.phenotype_mask.dtype

        def to_batch(v):
            v = torch.as_tensor(v, device=device, dtype=dtype)
            return v.unsqueeze(0) if v.dim() == 0 else v

        # Interpolation weights for each phenotype
        weights = {}
        batch_size = 1
        for name, val in [
            ("gender", gender),
            ("age", age),
            ("muscle", muscle),
            ("weight", weight),
            ("height", height),
            ("proportions", proportions),
            ("cupsize", cupsize),
            ("firmness", firmness),
        ]:
            val = to_batch(val)
            batch_size = max(batch_size, len(val))
            anchors = self._anchors[name]
            idx = torch.searchsorted(anchors, val, side="left").clamp(1, len(anchors) - 1)
            alpha = (val - anchors[idx - 1]) / (anchors[idx] - anchors[idx - 1])
            if not self.extrapolate_phenotypes:
                alpha = alpha.clamp(0, 1)
            w = torch.zeros(len(val), len(anchors), device=device, dtype=dtype)
            w.scatter_(1, (idx - 1).unsqueeze(1), (1 - alpha).unsqueeze(1))
            w.scatter_(1, idx.unsqueeze(1), alpha.unsqueeze(1))
            weights[name] = {k: w[:, i] for i, k in enumerate(PHENOTYPE_VARIATIONS[name])}

        # Race weights (normalized)
        race = torch.stack([to_batch(v) for v in (african, asian, caucasian)], dim=1)
        race = torch.nan_to_num(race / race.sum(dim=1, keepdim=True), 1 / 3, 1 / 3, 1 / 3)

        # Stack all phenotype weights
        all_weights = {k: v for d in weights.values() for k, v in d.items()}
        all_weights.update(african=race[:, 0], asian=race[:, 1], caucasian=race[:, 2])
        phens = torch.stack(
            [all_weights[k].expand(batch_size) for vs in PHENOTYPE_VARIATIONS.values() for k in vs], dim=1
        )

        # Compute blendshape coefficients via masked product
        masked = phens.unsqueeze(1) * self.phenotype_mask.unsqueeze(0)
        return torch.prod(masked + (1 - self.phenotype_mask.unsqueeze(0)), dim=-1)


# Conversion functions (native Z-up <-> API Y-up)


def from_native_args(
    pose: Float[Tensor, "B J 4 4"],
) -> dict[str, Tensor]:
    """Convert native ANNY args (4x4 transforms) to API format (axis-angle).

    Args:
        pose: Per-joint 4x4 rotation transforms [B, J, 4, 4] in Z-up coords

    Returns:
        Dict with 'pose' as axis-angle [B, J, 3]
    """
    R = pose[..., :3, :3]
    axis_angle = SO3.to_axis_angle(SO3.from_matrix(R))
    return {"pose": axis_angle}


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
) -> dict[str, Tensor]:
    """Convert API outputs (Y-up) to native ANNY format (Z-up).

    Args:
        vertices: Mesh vertices [B, V, 3] in Y-up coords
        transforms: Joint transforms [B, J, 4, 4] in Y-up coords

    Returns:
        Dict with 'vertices' and 'bone_poses' in Z-up coords
    """
    device, dtype = vertices.device, vertices.dtype
    coord_rot = _COORD_ROTATION.to(device=device, dtype=dtype)
    coord_trans = _COORD_TRANSLATION.to(device=device, dtype=dtype)

    # Inverse transform: Y-up -> Z-up
    # Forward was: v_yup = v_zup @ R.T + t
    # Inverse: v_zup = (v_yup - t) @ R
    native_verts = (vertices - coord_trans) @ coord_rot

    # For transforms: T_yup = coord @ T_zup
    # Inverse: T_zup = coord_inv @ T_yup
    coord_T = torch.eye(4, device=device, dtype=dtype)
    coord_T[:3, :3] = coord_rot
    coord_T[:3, 3] = coord_trans
    coord_T_inv = _invert_transform(coord_T)
    native_transforms = coord_T_inv @ transforms

    return {"vertices": native_verts, "bone_poses": native_transforms}


def _invert_transform(T: Float[Tensor, "*batch 4 4"]) -> Float[Tensor, "*batch 4 4"]:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_t = R.transpose(-1, -2)
    inv = torch.zeros_like(T)
    inv[..., :3, :3] = R_t
    inv[..., :3, 3] = -(R_t @ t.unsqueeze(-1)).squeeze(-1)
    inv[..., 3, 3] = 1.0
    return inv


# Kinematics


def _build_kinematic_fronts(parents: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """Group joints by depth for parallel forward kinematics."""
    n = len(parents)
    assigned = [False] * n
    level = [i for i in range(n) if parents[i] < 0]
    indices, parent_ids = [], []

    while level:
        indices.append(level)
        parent_ids.append([parents[i] for i in level])
        for j in level:
            assigned[j] = True
        level = [i for i in range(n) if not assigned[i] and parents[i] in level]

    return indices, parent_ids


def _forward_kinematics(
    fronts: tuple[list[list[int]], list[list[int]]],
    rest_poses: Float[Tensor, "B J 4 4"],
    delta_T: Float[Tensor, "B J 4 4"],
    base_T: Float[Tensor, "B 4 4"],
) -> tuple[Float[Tensor, "B J 4 4"], Float[Tensor, "B J 4 4"]]:
    """Parallel forward kinematics (autograd-compatible)."""
    B, J = rest_poses.shape[:2]

    T = rest_poses @ delta_T
    rest_inv = _invert_transform(rest_poses)

    poses = [None] * J
    transforms = [None] * J

    for joint_ids, parent_ids in zip(*fronts):
        roots = [(j, p) for j, p in zip(joint_ids, parent_ids) if p == -1]
        children_list = [(j, p) for j, p in zip(joint_ids, parent_ids) if p >= 0]

        if roots:
            root_ids = [j for j, _ in roots]
            root_poses = base_T[:, None] @ T[:, root_ids]
            root_transforms = root_poses @ rest_inv[:, root_ids]
            for idx, joint_id in enumerate(root_ids):
                poses[joint_id] = root_poses[:, idx]
                transforms[joint_id] = root_transforms[:, idx]

        if children_list:
            child_ids = [j for j, _ in children_list]
            parent_ids_list = [p for _, p in children_list]
            parent_transforms = torch.stack([transforms[p] for p in parent_ids_list], dim=1)
            child_poses = parent_transforms @ T[:, child_ids]
            child_transforms = child_poses @ rest_inv[:, child_ids]
            for idx, joint_id in enumerate(child_ids):
                poses[joint_id] = child_poses[:, idx]
                transforms[joint_id] = child_transforms[:, idx]

    poses_tensor = torch.stack(poses, dim=1)
    transforms_tensor = torch.stack(transforms, dim=1)
    return poses_tensor, transforms_tensor


def _bone_poses_from_heads_tails(
    heads: Float[Tensor, "B J 3"],
    tails: Float[Tensor, "B J 3"],
    rolls: Float[Tensor, "J 3 3"],
    y_axis: Float[Tensor, "3"],
    degen_rot: Float[Tensor, "3 3"],
    eps: float = 0.1,
) -> Float[Tensor, "B J 4 4"]:
    """Compute bone poses from head/tail positions."""
    vec = tails - heads
    y = vec / torch.linalg.norm(vec, dim=-1, keepdim=True)
    cross = torch.linalg.cross(y, y_axis.expand_as(y))
    dot = (y * y_axis).sum(dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1)

    axis = cross / cross_norm.unsqueeze(-1)
    angle = torch.atan2(cross_norm, dot)
    R = SO3.to_matrix(SO3.from_axis_angle(-angle.unsqueeze(-1) * axis))

    valid = (torch.abs((axis**2).sum(-1) - 1) < eps)[..., None, None]
    R = torch.where(valid, R, degen_rot.expand_as(R))
    R = R @ rolls

    H = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    H[..., :3, :3] = R
    H[..., :3, 3] = heads
    H[..., 3, 3] = 1.0
    return H


# Data loading

_RIG_CONFIGS = {
    "default": ("rig.default.json", "weights.default.json"),
    "default_no_toes": ("rig.default_no_toes.json", "weights.default.json"),
    "cmu_mb": ("rig.cmu_mb.json", "weights.cmu_mb.json"),
    "game_engine": ("rig.game_engine.json", "weights.game_engine.json"),
    "mixamo": ("rig.mixamo.json", "weights.mixamo.json"),
}


def _load_data(data_dir: Path, cache_dir: Path, rig: str, eyes: bool, tongue: bool) -> dict:
    """Load ANNY model data (cached)."""
    cache_key = hashlib.md5(f"{rig}_{eyes}_{tongue}".encode()).hexdigest()
    cache_file = cache_dir / f"data_{cache_key}.pth"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    dtype = torch.float64
    world_T = (
        0.1 * SO3.to_matrix(SO3.from_euler(torch.tensor([[torch.pi / 2, 0, 0]], dtype=dtype), convention="xyz"))[0]
    )

    # Load mesh
    mesh_path = data_dir / "data" / "mpfb2" / "3dobjs" / "base.obj"
    verts, uvs, groups = _load_obj(mesh_path, dtype)
    verts = verts @ world_T.T

    for g in groups.values():
        g["unique_verts"] = torch.unique(g["faces"].flatten())

    # Collect faces
    face_groups = ["body"] + (["helper-l-eye", "helper-r-eye"] if eyes else []) + (["helper-tongue"] if tongue else [])
    faces = torch.cat([groups[g]["faces"] for g in face_groups])
    face_uvs = torch.cat([groups[g]["face_uvs"] for g in face_groups])

    # Load rig
    rig_file, weights_file = _RIG_CONFIGS[rig]
    rig_dir = data_dir / "data" / "mpfb2" / "rigs" / "standard"
    rig_data = json.loads((rig_dir / rig_file).read_text())
    if rig == "mixamo":
        rig_data = rig_data["bones"]
    weights_data = json.loads((rig_dir / weights_file).read_text())

    bone_labels, bone_parents = _build_skeleton(rig_data)
    bone_indices, bone_weights = _build_skin_weights(weights_data, bone_labels, len(verts), dtype)

    # Load blendshapes
    blendshapes_dict = _load_blendshapes(data_dir, verts, world_T, dtype)
    blendshapes, phenotype_mask = _stack_blendshapes(blendshapes_dict, dtype)

    # Bone positions from blendshapes
    heads, tails, heads_bs, tails_bs, rolls = _compute_bone_data(
        verts, blendshapes, bone_labels, rig_data, groups, dtype
    )

    result = {
        "template_vertices": verts,
        "faces": faces,
        "face_uvs": face_uvs,
        "blendshapes": blendshapes,
        "phenotype_mask": phenotype_mask,
        "template_bone_heads": heads,
        "template_bone_tails": tails,
        "bone_heads_blendshapes": heads_bs,
        "bone_tails_blendshapes": tails_bs,
        "bone_rolls_rotmat": rolls,
        "bone_labels": bone_labels,
        "bone_parents": bone_parents,
        "vertex_bone_weights": bone_weights,
        "vertex_bone_indices": bone_indices,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_file)
    return result


def _load_obj(path: Path, dtype: torch.dtype) -> tuple[Tensor, Tensor, dict]:
    """Load OBJ file."""
    verts, uvs, groups, cur = [], [], {}, None

    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif cmd == "vt":
            uvs.append([float(x) for x in parts[1:3]])
        elif cmd == "g":
            cur = parts[1]
            groups[cur] = {"faces": [], "face_uvs": []}
        elif cmd == "f" and cur:
            fv, ft = [], []
            for p in parts[1:]:
                idx = p.split("/")
                fv.append(int(idx[0]) - 1)
                if len(idx) > 1 and idx[1]:
                    ft.append(int(idx[1]) - 1)
            groups[cur]["faces"].append(fv)
            if ft:
                groups[cur]["face_uvs"].append(ft)

    for g in groups.values():
        g["faces"] = torch.tensor(g["faces"], dtype=torch.int64)
        g["face_uvs"] = (
            torch.tensor(g["face_uvs"], dtype=torch.int64) if g["face_uvs"] else torch.empty(0, dtype=torch.int64)
        )

    return (
        torch.tensor(verts, dtype=dtype),
        torch.tensor(uvs, dtype=dtype) if uvs else torch.empty(0, 2, dtype=dtype),
        groups,
    )


def _load_blendshapes(
    data_dir: Path, template: Float[Tensor, "V 3"], world_T: Float[Tensor, "3 3"], dtype: torch.dtype
) -> dict:
    """Load all phenotype blendshapes."""
    n = len(template)
    pv = PHENOTYPE_VARIATIONS
    macro = data_dir / "data" / "mpfb2" / "targets" / "macrodetails"
    breast = data_dir / "data" / "mpfb2" / "targets" / "breast"
    newborn_scale = torch.tensor([0.922, 0.922, 0.75], dtype=dtype)

    def load(path: Path, age: str) -> Tensor:
        bs = torch.zeros(n, 3, dtype=dtype)
        with gzip.open(path, "rt") as f:
            for line in f:
                p = line.split()
                bs[int(p[0])] = torch.tensor([float(x) for x in p[1:]], dtype=dtype)
        bs = bs @ world_T.T
        if age == "newborn":
            bs = newborn_scale * bs + ((newborn_scale - 1) / 3) * template
        return bs

    def age_file(a):
        return "baby" if a == "newborn" else a

    shapes = {}
    for g, a, m, w in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"]):
        shapes[(g, a, m, w)] = load(macro / f"universal-{g}-{age_file(a)}-{m}-{w}.target.gz", a)
    for r, g, a in itertools.product(pv["race"], pv["gender"], pv["age"]):
        shapes[(r, g, a)] = load(macro / f"{r}-{g}-{age_file(a)}.target.gz", a)
    for g, a, m, w, h in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"], pv["height"]):
        shapes[(g, a, m, w, h)] = load(macro / "height" / f"{g}-{age_file(a)}-{m}-{w}-{h}.target.gz", a)
    for g, a, m, w, p in itertools.product(pv["gender"], pv["age"], pv["muscle"], pv["weight"], pv["proportions"]):
        if a not in ("newborn", "baby"):
            shapes[(g, a, m, w, p)] = load(macro / "proportions" / f"{g}-{a}-{m}-{w}-{p}.target.gz", a)
    for a, m, w, c, f in itertools.product(pv["age"], pv["muscle"], pv["weight"], pv["cupsize"], pv["firmness"]):
        path = breast / f"female-{a}-{m}-{w}-{c}-{f}.target.gz"
        if path.exists():
            shapes[("female", a, m, w, c, f)] = load(path, a)

    return shapes


def _stack_blendshapes(shapes: dict, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Stack blendshapes and build phenotype mask."""
    all_phens = [p for vs in PHENOTYPE_VARIATIONS.values() for p in vs]
    stacked, masks = [], []
    for keys, shape in shapes.items():
        stacked.append(shape)
        mask = torch.zeros(len(all_phens), dtype=dtype)
        for k in keys:
            mask[all_phens.index(k)] = 1
        masks.append(mask)
    return torch.stack(stacked), torch.stack(masks)


def _build_skeleton(rig_data: dict) -> tuple[list[str], list[int]]:
    """Build bone hierarchy from rig data."""
    root = next(k for k, v in rig_data.items() if v["parent"] == "")
    labels, parents = [], []

    def add(label, parent_id):
        labels.append(label)
        parents.append(parent_id)
        cur_id = len(labels) - 1
        for k, v in rig_data.items():
            if k not in labels and v["parent"] == label:
                add(k, cur_id)

    add(root, -1)
    return labels, parents


def _build_skin_weights(
    weights_data: dict, labels: list[str], n_verts: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Build sparse skin weight tensors."""
    indices = [[] for _ in range(n_verts)]
    weights = [[] for _ in range(n_verts)]

    for bone_id, label in enumerate(labels):
        for v_idx, w in weights_data["weights"][label]:
            indices[v_idx].append(bone_id)
            weights[v_idx].append(w)

    max_k = max(len(i) for i in indices)
    for i, w in zip(indices, weights):
        while len(i) < max_k:
            i.append(0)
            w.append(0.0)

    idx_t = torch.tensor(indices, dtype=torch.int64)
    w_t = torch.tensor(weights, dtype=dtype)
    w_t /= w_t.sum(dim=-1, keepdim=True)
    return idx_t, w_t


def _compute_bone_data(
    verts: Float[Tensor, "V 3"],
    shapes: Float[Tensor, "S V 3"],
    labels: list,
    rig: dict,
    groups: dict,
    dtype: torch.dtype,
):
    """Compute bone head/tail positions and blendshapes."""

    def get_verts(data):
        s = data["strategy"]
        if s == "VERTEX":
            return [data["vertex_index"]]
        if s == "CUBE":
            return groups[data["cube_name"]]["unique_verts"].tolist()
        if s == "MEAN":
            return data["vertex_indices"]
        raise ValueError(s)

    heads, tails, heads_bs, tails_bs, rolls = [], [], [], [], []
    for label in labels:
        h_idx = torch.tensor(get_verts(rig[label]["head"]), dtype=torch.int64)
        t_idx = torch.tensor(get_verts(rig[label]["tail"]), dtype=torch.int64)
        heads.append(verts[h_idx].mean(0))
        tails.append(verts[t_idx].mean(0))
        heads_bs.append(shapes[:, h_idx].mean(1))
        tails_bs.append(shapes[:, t_idx].mean(1))
        rolls.append(rig[label]["roll"])

    euler = torch.zeros(len(rolls), 3, dtype=dtype)
    euler[:, 1] = torch.tensor(rolls, dtype=dtype)
    rolls_mat = SO3.to_matrix(SO3.from_euler(euler, convention="xyz"))

    return torch.stack(heads), torch.stack(tails), torch.stack(heads_bs, dim=1), torch.stack(tails_bs, dim=1), rolls_mat


def _edit_mesh_faces(
    faces: Int[Tensor, "F 4"], uvs: Int[Tensor, "F 4"]
) -> tuple[Int[Tensor, "F2 4"], Int[Tensor, "F2 4"]]:
    """Edit MakeHuman mesh topology (ear caps)."""
    discard = torch.cat([torch.arange(1778, 1794), torch.arange(8450, 8466)]).to(faces.dtype)
    keep = ~torch.isin(faces, discard).any(dim=1)

    v2uv = {}
    for i in torch.nonzero(~keep, as_tuple=False).squeeze(1):
        for v, uv in zip(faces[i], uvs[i]):
            v2uv[v.item()] = uv.item()

    caps = torch.tensor(
        [
            [8437, 8438, 8439, 8440],
            [8436, 8437, 8440, 8441],
            [8435, 8436, 8441, 8442],
            [8434, 8435, 8442, 8443],
            [8449, 8434, 8443, 8444],
            [8448, 8449, 8444, 8445],
            [8447, 8448, 8445, 8446],
            [1762, 1771, 1770, 1763],
            [1763, 1770, 1769, 1764],
            [1764, 1769, 1768, 1765],
            [1765, 1768, 1767, 1766],
            [1762, 1777, 1772, 1771],
            [1777, 1776, 1773, 1772],
            [1776, 1775, 1774, 1773],
        ],
        dtype=faces.dtype,
    )
    cap_uvs = torch.tensor([v2uv[v.item()] for v in caps.flatten()]).reshape_as(caps)

    return torch.cat([faces[keep], caps]), torch.cat([uvs[keep], cap_uvs])


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
