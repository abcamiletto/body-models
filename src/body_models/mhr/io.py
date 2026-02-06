"""I/O utilities for MHR model loading."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from .. import config
from ..utils import download_and_extract, get_cache_dir

MHR_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve MHR model path, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("mhr")

    if model_path is not None:
        model_path = Path(model_path)
        if (model_path / "mhr_model.pt").exists():
            return model_path
        if model_path.exists():
            return model_path.parent
        raise FileNotFoundError(f"MHR model path {model_path} does not exist")

    cache_path = get_cache_dir() / "mhr"
    if (cache_path / "mhr_model.pt").exists():
        return cache_path

    return download_model()


def download_model() -> Path:
    """Download MHR model from GitHub releases."""
    cache_dir = get_cache_dir() / "mhr"
    print(f"Downloading MHR model to {cache_dir}...")
    download_and_extract(url=MHR_URL, dest=cache_dir, extract_subdir="assets/")
    print("Done")
    return cache_dir


def load_model_data(asset_dir: Path) -> dict[str, Tensor]:
    """Load MHR model data from disk."""
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
    """Build dense skinning matrices from sparse representation."""
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


def compute_kinematic_fronts(parents: Int[Tensor, "J"] | np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK. Returns [(joint_indices, parent_indices), ...]."""
    parents_list = parents.tolist()  # Works for both numpy arrays and torch tensors

    n_joints = len(parents_list)
    processed: set[int] = set()
    fronts: list[tuple[list[int], list[int]]] = []

    while len(processed) < n_joints:
        joints: list[int] = []
        joint_parents: list[int] = []
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


def simplify_mesh(
    vertices: Float[np.ndarray, "V 3"],
    faces: Int[np.ndarray, "F 3"],
    target_faces: int,
) -> tuple[Float[np.ndarray, "V2 3"], Int[np.ndarray, "F2 3"], Int[np.ndarray, "V2"]]:
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


# ============================================================================
# Pose correctives (PyTorch only)
# ============================================================================


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


def load_pose_correctives(asset_dir: Path, lod: int) -> _PoseCorrectivesModel:
    """Load neural pose correctives model (PyTorch only)."""
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
