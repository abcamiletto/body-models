"""I/O utilities for MHR model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .. import config
from ..common import simplify_mesh
from ..utils import download_and_extract, get_cache_dir

__all__ = [
    "get_model_path",
    "download_model",
    "load_model_data",
    "compute_kinematic_fronts",
    "simplify_mesh",
    "load_pose_correctives_weights",
    "load_pose_correctives",
]

MHR_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"


class MHRModelData(TypedDict):
    base_vertices: np.ndarray
    blendshape_dirs: np.ndarray
    joint_parents: np.ndarray
    joint_names: list[str]
    joint_offsets: np.ndarray
    joint_pre_rotations: np.ndarray
    skin_weights: np.ndarray
    skin_indices: np.ndarray
    parameter_transform: np.ndarray
    inverse_bind_pose: np.ndarray
    faces: np.ndarray


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


def load_model_data(asset_dir: Path) -> MHRModelData:
    """Load MHR model data from disk without requiring torch."""
    model = _load_checkpoint_numpy(asset_dir / "mhr_model.pt")
    character = _get_attr(model, "character_torch")
    skeleton = character.skeleton
    lbs = character.linear_blend_skinning
    blend_shape = character.blend_shape

    skin_indices, skin_weights = _build_dense_skinning(
        lbs.vert_indices_flattened,
        lbs.skin_indices_flattened,
        lbs.skin_weights_flattened,
        blend_shape.base_shape.shape[0],
    )

    return {
        "base_vertices": blend_shape.base_shape,
        "blendshape_dirs": np.concatenate(
            [
                blend_shape.shape_vectors,
                model.face_expressions_model.shape_vectors,
            ],
            axis=0,
        ),
        "joint_parents": skeleton.joint_parents,
        "joint_names": [str(x) for x in skeleton.joint_names],
        "joint_offsets": skeleton.joint_translation_offsets,
        "joint_pre_rotations": skeleton.joint_prerotations,
        "skin_weights": skin_weights,
        "skin_indices": skin_indices,
        "parameter_transform": character.parameter_transform.parameter_transform,
        "inverse_bind_pose": lbs.inverse_bind_pose,
        "faces": character.mesh.faces,
    }


def _get_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def _load_checkpoint_numpy(checkpoint_path: Path) -> Any:
    try:
        from ptloader import CheckpointError, load
    except ImportError as exc:
        raise ImportError("ptloader is required to load MHR checkpoints without torch.") from exc

    try:
        return load(checkpoint_path, weights_only=True)
    except (CheckpointError, ValueError):
        # Keep one fallback with explicit permissive mode for older ptloader variants.
        return load(checkpoint_path, weights_only=True, torchscript_mode="permissive")


def _build_dense_skinning(
    vert_indices: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    num_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense skinning matrices from sparse representation."""
    vert_indices = vert_indices.astype(np.int64, copy=False)
    joint_indices = joint_indices.astype(np.int64, copy=False)
    counts = np.bincount(vert_indices, minlength=num_vertices)
    K = int(counts.max())

    dense_indices = np.zeros((num_vertices, K), dtype=np.int64)
    dense_weights = np.zeros((num_vertices, K), dtype=joint_weights.dtype)

    offsets = np.zeros(num_vertices + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    for v in range(num_vertices):
        start, end = int(offsets[v]), int(offsets[v + 1])
        dense_indices[v, : end - start] = joint_indices[start:end]
        dense_weights[v, : end - start] = joint_weights[start:end]

    return dense_indices, dense_weights


def compute_kinematic_fronts(parents: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Compute kinematic fronts for batched FK. Returns [(joint_indices, parent_indices), ...]."""
    parents_list = parents.tolist()

    n_joints = len(parents_list)
    processed: set[int] = set()
    fronts: list[tuple[list[int], list[int]]] = []

    while len(processed) < n_joints:
        joints: list[int] = []
        joint_parents: list[int] = []
        for j in range(n_joints):
            if j in processed:
                continue
            p = int(parents_list[j])
            if p < 0 or p in processed:
                joints.append(j)
                joint_parents.append(p)
        fronts.append((joints, joint_parents))
        processed.update(joints)

    return fronts


def load_pose_correctives_weights(asset_dir: Path, lod: int) -> dict[str, np.ndarray]:
    """Load pose correctives weights as numpy arrays (backend-agnostic).

    Args:
        asset_dir: Path to MHR assets directory.
        lod: Level of detail (1 = default).

    Returns:
        Dict with 'W1' [3000, 750] and 'W2' [V*3, 3000] weight matrices.
    """
    blend_data = dict(np.load(asset_dir / f"corrective_blendshapes_lod{lod}.npz"))
    act_data = dict(np.load(asset_dir / "corrective_activation.npz"))

    sparse_indices = act_data["0.sparse_indices"]
    sparse_weight = act_data["0.sparse_weight"]

    out_features, in_features = 125 * 24, 125 * 6
    W1 = np.zeros((out_features, in_features), dtype=np.float32)
    W1[sparse_indices[0], sparse_indices[1]] = sparse_weight

    corrective_blendshapes = blend_data["corrective_blendshapes"]
    n_comp = corrective_blendshapes.shape[0]
    W2 = corrective_blendshapes.reshape(n_comp, -1).T.astype(np.float32)

    return {"W1": W1, "W2": W2}


def load_pose_correctives(asset_dir: Path, lod: int) -> Any:
    """Load neural pose correctives model (PyTorch nn.Module).

    .. deprecated::
        Use :func:`load_pose_correctives_weights` for backend-agnostic weights.
    """
    import torch
    import torch.nn as nn
    from nanomanifold import SO3
    from torch import Tensor

    class _SparseLinear(nn.Module):
        sparse_weight: Tensor
        dense_weight: Tensor
        _sparse_indices: Tensor

        def __init__(self, in_features: int, out_features: int, sparse_mask: np.ndarray) -> None:
            super().__init__()
            idx = torch.from_numpy(sparse_mask).nonzero().T
            self.register_buffer("_sparse_indices", idx, persistent=False)
            self.sparse_indices = nn.Parameter(idx, requires_grad=False)
            self.sparse_weight = nn.Parameter(torch.zeros(idx.shape[1]), requires_grad=False)
            self.register_buffer("dense_weight", torch.zeros(out_features, in_features), persistent=False)
            self._weight_initialized = False

        def _ensure_dense_weight(self) -> None:
            if not self._weight_initialized:
                self.dense_weight[self._sparse_indices[0], self._sparse_indices[1]] = self.sparse_weight
                self._weight_initialized = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._ensure_dense_weight()
            return x @ self.dense_weight.T

    class _PoseCorrectivesModel(nn.Module):
        def __init__(self, predictor: nn.Sequential) -> None:
            super().__init__()
            self.predictor = predictor

        def forward(self, joint_params: torch.Tensor) -> torch.Tensor:
            euler = joint_params[:, 2:, 3:6]
            rot = SO3.conversions.from_euler_to_matrix(euler, convention="xyz", xp=torch)
            feat = torch.cat([rot[..., 0], rot[..., 1]], dim=-1)
            feat[:, :, 0] -= 1
            feat[:, :, 4] -= 1
            corr = self.predictor(feat.flatten(1, 2))
            return corr.reshape(joint_params.shape[0], -1, 3)

    blend_data = dict(np.load(asset_dir / f"corrective_blendshapes_lod{lod}.npz"))
    act_data = dict(np.load(asset_dir / "corrective_activation.npz"))

    n_comp, n_v = blend_data["corrective_blendshapes"].shape[:2]
    predictor = nn.Sequential(
        _SparseLinear(125 * 6, 125 * 24, act_data["posedirs_sparse_mask"]),
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
