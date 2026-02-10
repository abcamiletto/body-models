"""I/O utilities for MHR model loading."""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

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
]

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

    # Build dense W1 from sparse representation
    sparse_indices = act_data["0.sparse_indices"]  # (2, N)
    sparse_weight = act_data["0.sparse_weight"]  # (N,)

    out_features, in_features = 125 * 24, 125 * 6  # 3000, 750
    W1 = np.zeros((out_features, in_features), dtype=np.float32)
    W1[sparse_indices[0], sparse_indices[1]] = sparse_weight

    # W2 from corrective_blendshapes
    corrective_blendshapes = blend_data["corrective_blendshapes"]  # (n_comp, n_v, 3)
    n_comp = corrective_blendshapes.shape[0]
    W2 = corrective_blendshapes.reshape(n_comp, -1).T.astype(np.float32)  # (V*3, n_comp)

    return {"W1": W1, "W2": W2}
