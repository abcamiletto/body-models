# Derived from: https://github.com/facebookresearch/MHR
# Original license: Apache 2.0 (https://github.com/facebookresearch/MHR/blob/main/LICENSE)

import math
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from .io import get_model_path

_LN2 = math.log(2)


class MHR(BodyModel):
    """MHR body model with neural pose correctives.

    Args:
        model_path: Path to MHR model directory. Auto-downloads if None.
        lod: Level of detail (1 = default).
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
            Note: Neural pose correctives are computed at full resolution, then downsampled.

    Forward API:
        forward_vertices(shape, pose, expression, global_rotation, global_translation)
        forward_skeleton(shape, pose, expression, global_rotation, global_translation)

        shape: [B, 45] identity blendshapes
        pose: [B, 204] joint parameters
        expression: [B, 72] facial blendshapes (optional)
    """

    SHAPE_DIM = 45
    EXPR_DIM = 72
    _PARAMS_PER_JOINT = 7

    # Full-resolution buffers (for pose correctives)
    base_vertices_full: Float[Tensor, "V_full 3"]
    blendshape_dirs_full: Float[Tensor, "117 V_full 3"]
    _skin_weights_full: Float[Tensor, "V_full K"]
    _skin_indices_full: Int[Tensor, "V_full K"]

    # Output buffers (simplified or same as full)
    base_vertices: Float[Tensor, "V 3"]
    blendshape_dirs: Float[Tensor, "117 V 3"]
    _skin_weights: Float[Tensor, "V K"]
    _skin_indices: Int[Tensor, "V K"]
    _vertex_map: Int[Tensor, "V"] | None

    # Skeleton/other buffers
    joint_parents: Int[Tensor, "J"]
    joint_offsets: Float[Tensor, "J 3"]
    joint_pre_rotations: Float[Tensor, "J 4"]
    parameter_transform: Float[Tensor, "_ _"]
    _faces: Int[Tensor, "F 3"]
    bind_inv_linear: Float[Tensor, "J 3 3"]
    bind_inv_translation: Float[Tensor, "J 3"]

    def __init__(self, model_path: Path | str | None = None, *, lod: int = 1, simplify: float = 1.0) -> None:
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

        resolved_path = get_model_path(model_path)
        model = _load_model_data(resolved_path)

        base_vertices_full = model["base_vertices"]
        blendshape_dirs_full = model["blendshape_dirs"]
        skin_weights_full = model["skin_weights"]
        skin_indices_full = model["skin_indices"].to(torch.int64)
        faces = model["faces"]

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            vertices_np = base_vertices_full.numpy()
            faces_np = faces.numpy().astype(int)
            new_vertices, new_faces, vertex_map = _simplify_mesh(vertices_np, faces_np, target_faces)

            # Simplified output buffers
            self.register_buffer("base_vertices", torch.as_tensor(new_vertices, dtype=base_vertices_full.dtype))
            self.register_buffer(
                "blendshape_dirs", blendshape_dirs_full[:, vertex_map].clone()
            )
            self.register_buffer("_skin_weights", skin_weights_full[vertex_map].clone())
            self.register_buffer("_skin_indices", skin_indices_full[vertex_map].clone())
            self.register_buffer("_faces", torch.as_tensor(new_faces, dtype=torch.int64))
            self.register_buffer("_vertex_map", torch.as_tensor(vertex_map, dtype=torch.int64))
        else:
            self.register_buffer("base_vertices", base_vertices_full)
            self.register_buffer("blendshape_dirs", blendshape_dirs_full)
            self.register_buffer("_skin_weights", skin_weights_full)
            self.register_buffer("_skin_indices", skin_indices_full)
            self.register_buffer("_faces", faces)
            self._vertex_map = None

        # Full-resolution buffers for pose correctives (always needed)
        self.register_buffer("base_vertices_full", base_vertices_full)
        self.register_buffer("blendshape_dirs_full", blendshape_dirs_full)
        self.register_buffer("_skin_weights_full", skin_weights_full)
        self.register_buffer("_skin_indices_full", skin_indices_full)

        # Skeleton buffers
        self.register_buffer("joint_parents", model["joint_parents"].to(torch.int32))
        self.register_buffer("joint_offsets", model["joint_offsets"])
        self.register_buffer("joint_pre_rotations", model["joint_pre_rotations"])
        self.register_buffer("parameter_transform", model["parameter_transform"])

        inv_bind = model["inverse_bind_pose"]
        t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
        self.register_buffer("bind_inv_linear", SO3.to_matrix(q, xyzw=True) * s.unsqueeze(-1))
        self.register_buffer("bind_inv_translation", t)

        self._pose_correctives = _load_pose_correctives(resolved_path, lod)

        # Precompute kinematic fronts for batched FK
        self._kinematic_fronts = _compute_kinematic_fronts(self.joint_parents)

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.joint_offsets.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        V, K = self._skin_weights.shape
        dense = torch.zeros(V, self.num_joints, device=self._skin_weights.device, dtype=self._skin_weights.dtype)
        dense.scatter_(1, self._skin_indices, self._skin_weights)
        return dense

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.base_vertices * 0.01

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3] in meters."""
        B = pose.shape[0]
        if expression is None:
            expression = torch.zeros((B, self.EXPR_DIM), device=pose.device, dtype=pose.dtype)

        vertices, _, _, _ = self._forward(shape, expression, pose, apply_correctives=True)
        vertices = vertices * 0.01

        if global_rotation is not None:
            R = SO3.to_matrix(SO3.from_axis_angle(global_rotation))
            vertices = vertices @ R.mT
        if global_translation is not None:
            vertices = vertices + global_translation[:, None]
        return vertices

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute skeleton transforms [B, J, 4, 4] in meters."""
        t_g, r_g, s_g = self._forward_skeleton_core(pose)
        T = self._trs_to_transforms(t_g * 0.01, r_g, s_g)

        if global_rotation is not None or global_translation is not None:
            B = T.shape[0]
            global_T = torch.eye(4, device=T.device, dtype=T.dtype).expand(B, 4, 4).clone()
            if global_rotation is not None:
                global_T[:, :3, :3] = SO3.to_matrix(SO3.from_axis_angle(global_rotation))
            if global_translation is not None:
                global_T[:, :3, 3] = global_translation
            T = global_T[:, None] @ T
        return T

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.base_vertices.device
        return {
            "shape": torch.zeros((1, self.SHAPE_DIM), device=device, dtype=dtype),
            "pose": torch.zeros((batch_size, self.pose_dim), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, self.EXPR_DIM), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }

    def _forward(
        self,
        shape: Float[Tensor, "B 45"],
        expression: Float[Tensor, "B 72"],
        pose: Float[Tensor, "B 204"],
        apply_correctives: bool = True,
    ) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"]]:
        """Full forward pass returning (vertices, t_g, r_g, s_g)."""
        B = pose.shape[0]
        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)
        if expression.shape[0] == 1 and B > 1:
            expression = expression.expand(B, -1)
        coeffs = torch.cat([shape, expression], dim=1)

        t_g, r_g, s_g, j_p = self._forward_skeleton_core(pose, return_joint_params=True)

        # Pose correctives must be computed at full resolution
        if apply_correctives and self._vertex_map is not None:
            # Compute full-resolution v_t with correctives
            v_t_full = self.base_vertices_full + torch.einsum("bi,ivk->bvk", coeffs, self.blendshape_dirs_full)
            v_t_full = v_t_full + self._pose_correctives(j_p)
            # Downsample to simplified vertices
            v_t = v_t_full[:, self._vertex_map]
        else:
            # No simplification or no correctives - use direct computation
            v_t = self.base_vertices + torch.einsum("bi,ivk->bvk", coeffs, self.blendshape_dirs)
            if apply_correctives:
                v_t = v_t + self._pose_correctives(j_p)

        # Linear blend skinning - use r_g directly (no quaternion conversion needed)
        lin_g = r_g * s_g.unsqueeze(-1)
        lin = lin_g @ self.bind_inv_linear
        t = torch.einsum("bjik,jk->bji", lin_g, self.bind_inv_translation) + t_g

        lin = lin[:, self._skin_indices]
        t = t[:, self._skin_indices]
        v_transformed = torch.einsum("bvkij,bvj->bvki", lin, v_t) + t
        verts = (v_transformed * self._skin_weights[None, :, :, None]).sum(dim=2)

        return verts, t_g, r_g, s_g

    @overload
    def _forward_skeleton_core(
        self, pose: Float[Tensor, "B 204"], return_joint_params: Literal[False] = ...
    ) -> tuple[Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"]]: ...

    @overload
    def _forward_skeleton_core(
        self, pose: Float[Tensor, "B 204"], return_joint_params: Literal[True]
    ) -> tuple[Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"], Float[Tensor, "B J 7"]]: ...

    def _forward_skeleton_core(
        self, pose: Float[Tensor, "B 204"], return_joint_params: bool = False
    ) -> (
        tuple[Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"]]
        | tuple[Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"], Float[Tensor, "B J 7"]]
    ):
        """Compute global skeleton transforms from pose.

        Returns (t_g, r_g, s_g[, j_p]) - translations, rotation matrices, scales.
        """
        j_p = self._pose_to_joint_params(pose)

        t_l = j_p[..., :3] + self.joint_offsets
        q_l = SO3.to_quat_xyzw(SO3.canonicalize(SO3.from_euler(j_p[..., 3:6], convention="xyz")))
        q_l = SO3.canonicalize(SO3.multiply(self.joint_pre_rotations, q_l, xyzw=True), xyzw=True)
        s_l = torch.exp(_LN2 * j_p[..., 6:7])

        t_g, r_g, s_g = self._compose_global_trs(t_l, q_l, s_l)

        if return_joint_params:
            return t_g, r_g, s_g, j_p
        return t_g, r_g, s_g

    def _pose_to_joint_params(self, pose: Float[Tensor, "B 204"]) -> Float[Tensor, "B J 7"]:
        """Convert pose vector to per-joint parameters [B, J, 7]."""
        pad = torch.zeros((pose.shape[0], self.SHAPE_DIM), device=pose.device, dtype=pose.dtype)
        j_p = torch.einsum("dn,bn->bd", self.parameter_transform, torch.cat([pose, pad], dim=-1))
        return j_p.reshape(pose.shape[0], self.num_joints, self._PARAMS_PER_JOINT)

    def _compose_global_trs(
        self,
        t_l: Float[Tensor, "B J 3"],
        q_l: Float[Tensor, "B J 4"],
        s_l: Float[Tensor, "B J 1"],
    ) -> tuple[Float[Tensor, "B J 3"], Float[Tensor, "B J 3 3"], Float[Tensor, "B J 1"]]:
        """Compose local TRS transforms into global via batched FK using kinematic fronts.

        Returns (t_g, r_g, s_g) - global translations, rotation matrices, and scales.
        Skips quaternion computation in the loop for efficiency.
        """
        B, J = t_l.shape[:2]
        r_l = SO3.to_matrix(q_l, xyzw=True)

        # Collect results per joint (avoids in-place ops for gradient compatibility)
        t_results: list[Tensor] = [torch.empty(0)] * J
        s_results: list[Tensor] = [torch.empty(0)] * J
        r_results: list[Tensor] = [torch.empty(0)] * J

        for joints, parents in self._kinematic_fronts:
            if parents[0] < 0:  # Root joints
                for j in joints:
                    t_results[j] = t_l[:, j]
                    s_results[j] = s_l[:, j]
                    r_results[j] = r_l[:, j]
            else:
                for j, p in zip(joints, parents):
                    r_results[j] = r_results[p] @ r_l[:, j]
                    s_results[j] = s_results[p] * s_l[:, j]
                    r_ps = r_results[p] * s_results[p][:, :, None]
                    t_results[j] = (r_ps @ t_l[:, j, :, None]).squeeze(-1) + t_results[p]

        t_g = torch.stack(t_results, dim=1)
        s_g = torch.stack(s_results, dim=1)
        r_g = torch.stack(r_results, dim=1)

        return t_g, r_g, s_g

    def _trs_to_transforms(
        self,
        t: Float[Tensor, "B J 3"],
        r: Float[Tensor, "B J 3 3"],
        s: Float[Tensor, "B J 1"],
    ) -> Float[Tensor, "B J 4 4"]:
        """Convert translation, rotation matrix, scale to 4x4 transforms."""
        R = r * s.unsqueeze(-1)
        B, J = t.shape[:2]
        T = torch.zeros(B, J, 4, 4, device=t.device, dtype=t.dtype)
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        T[..., 3, 3] = 1.0
        return T


class _SparseLinear(nn.Module):
    """Sparse linear layer for pose correctives."""

    dense_weight: Float[Tensor, "O I"]
    _sparse_indices: Int[Tensor, "2 N"]

    def __init__(self, in_features: int, out_features: int, sparse_mask: Float[Tensor, "O I"]) -> None:
        super().__init__()
        idx = sparse_mask.nonzero().T
        self.register_buffer("_sparse_indices", idx, persistent=False)
        self.sparse_indices = nn.Parameter(idx, requires_grad=False)
        self.sparse_weight = nn.Parameter(torch.zeros(idx.shape[1]), requires_grad=False)
        self.register_buffer("dense_weight", torch.zeros(out_features, in_features), persistent=False)
        self._weight_initialized = False

    def _ensure_dense_weight(self) -> None:
        """Lazily initialize dense weight from sparse representation."""
        if not self._weight_initialized:
            self.dense_weight[self._sparse_indices[0], self._sparse_indices[1]] = self.sparse_weight
            self._weight_initialized = True

    def forward(self, x: Float[Tensor, "B I"]) -> Float[Tensor, "B O"]:
        self._ensure_dense_weight()
        return x @ self.dense_weight.T


class _PoseCorrectivesModel(nn.Module):
    """Neural pose correctives predictor."""

    def __init__(self, predictor: nn.Sequential) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, joint_params: Float[Tensor, "B J 7"]) -> Float[Tensor, "B V 3"]:
        euler = joint_params[:, 2:, 3:6]
        rot = SO3.to_matrix(SO3.from_euler(euler, convention="xyz"))
        feat = torch.cat([rot[..., 0], rot[..., 1]], dim=-1)
        feat[:, :, 0] -= 1
        feat[:, :, 4] -= 1
        corr = self.predictor(feat.flatten(1, 2))
        return corr.reshape(joint_params.shape[0], -1, 3)


def _load_pose_correctives(asset_dir: Path, lod: int) -> _PoseCorrectivesModel:
    blend_data = dict(np.load(asset_dir / f"corrective_blendshapes_lod{lod}.npz"))
    act_data = dict(np.load(asset_dir / "corrective_activation.npz"))

    n_comp, n_v = blend_data["corrective_blendshapes"].shape[:2]

    predictor = nn.Sequential(
        _SparseLinear(125 * 6, 125 * 24, torch.from_numpy(act_data["posedirs_sparse_mask"])),
        nn.ReLU(),
        nn.Linear(125 * 24, n_v * 3, bias=False),
    )
    predictor.load_state_dict(
        {
            "0.sparse_indices": torch.from_numpy(act_data["0.sparse_indices"]),
            "0.sparse_weight": torch.from_numpy(act_data["0.sparse_weight"]),
            "2.weight": torch.from_numpy(blend_data["corrective_blendshapes"].reshape(n_comp, -1).T),
        }
    )
    for p in predictor.parameters():
        p.requires_grad = False

    model = _PoseCorrectivesModel(predictor)
    model.eval()
    return model


def _load_model_data(asset_dir: Path) -> dict[str, Tensor]:
    state = torch.jit.load(asset_dir / "mhr_model.pt").state_dict()

    skin_indices, skin_weights = _build_dense_skinning(
        state["character_torch.linear_blend_skinning.vert_indices_flattened"],
        state["character_torch.linear_blend_skinning.skin_indices_flattened"],
        state["character_torch.linear_blend_skinning.skin_weights_flattened"],
        state["character_torch.blend_shape.base_shape"].shape[0],
    )

    return {
        "base_vertices": state["character_torch.blend_shape.base_shape"],
        "blendshape_dirs": torch.cat(
            [
                state["character_torch.blend_shape.shape_vectors"],
                state["face_expressions_model.shape_vectors"],
            ],
            dim=0,
        ),
        "joint_parents": state["character_torch.skeleton.joint_parents"],
        "joint_offsets": state["character_torch.skeleton.joint_translation_offsets"],
        "joint_pre_rotations": state["character_torch.skeleton.joint_prerotations"],
        "skin_weights": skin_weights,
        "skin_indices": skin_indices,
        "parameter_transform": state["character_torch.parameter_transform.parameter_transform"],
        "inverse_bind_pose": state["character_torch.linear_blend_skinning.inverse_bind_pose"],
        "faces": state["character_torch.mesh.faces"],
    }


def _build_dense_skinning(
    vert_indices: Int[Tensor, "N"],
    joint_indices: Int[Tensor, "N"],
    joint_weights: Float[Tensor, "N"],
    num_vertices: int,
) -> tuple[Int[Tensor, "V K"], Float[Tensor, "V K"]]:
    counts = torch.bincount(vert_indices.to(torch.int64), minlength=num_vertices)
    K = int(counts.max().item())

    dense_indices = torch.zeros((num_vertices, K), dtype=torch.int64)
    dense_weights = torch.zeros((num_vertices, K), dtype=joint_weights.dtype)

    offsets = torch.zeros(num_vertices + 1, dtype=torch.int64)
    offsets[1:] = counts.cumsum(0)

    for v in range(num_vertices):
        start, end = int(offsets[v].item()), int(offsets[v + 1].item())
        dense_indices[v, : end - start] = joint_indices[start:end].to(torch.int64)
        dense_weights[v, : end - start] = joint_weights[start:end]

    return dense_indices, dense_weights


def _compute_kinematic_fronts(parents: Int[Tensor, "J"]) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK. Returns [(joint_indices, parent_indices), ...]."""
    parents_list = parents.tolist()
    n_joints = len(parents_list)
    processed = set()
    fronts = []

    while len(processed) < n_joints:
        joints, joint_parents = [], []
        for j in range(n_joints):
            if j in processed:
                continue
            p = parents_list[j]
            if p < 0 or p in processed:
                joints.append(j)
                joint_parents.append(p)
        fronts.append((joints, joint_parents))
        processed.update(joints)

    return fronts


def extract_skeleton_state(
    transforms: Float[Tensor, "B J 4 4"],
) -> Float[Tensor, "B J 8"]:
    """Extract skeleton state [t, q, s] from 4x4 transforms.

    Args:
        transforms: World-space 4x4 transform matrices [B, J, 4, 4].

    Returns:
        Skeleton state [B, J, 8] with [translation(3), quaternion_xyzw(4), scale(1)].
    """
    t = transforms[..., :3, 3]
    R = transforms[..., :3, :3]

    # Extract scale (uniform, from first column norm)
    s = torch.linalg.norm(R[..., :, 0], dim=-1, keepdim=True)

    # Extract pure rotation and convert to quaternion
    R_pure = R / s.unsqueeze(-1)
    q = SO3.to_quat_xyzw(SO3.from_matrix(R_pure))

    return torch.cat([t, q, s], dim=-1)


def from_native_args(
    shape: Float[Tensor, "B 45"],
    expression: Float[Tensor, "B 72"],
    pose: Float[Tensor, "B 204"],
) -> dict[str, Tensor]:
    """Convert native MHR args (shape, expression, pose) to forward_* kwargs."""
    return {"shape": shape, "pose": pose, "expression": expression}


def to_native_outputs(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
) -> dict[str, Tensor]:
    """Convert forward_* outputs to native MHR format (cm units, skeleton state).

    Args:
        vertices: Mesh vertices [B, V, 3] in meters.
        transforms: Skeleton transforms [B, J, 4, 4] in meters.

    Returns:
        Dict with "vertices" and "joints" in cm units.
    """
    skel_state = extract_skeleton_state(transforms)
    skel_state[..., :3] *= 100  # meters to cm
    return {
        "vertices": vertices * 100,
        "joints": skel_state,
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
