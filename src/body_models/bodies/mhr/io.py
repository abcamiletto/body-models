"""I/O utilities for MHR model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from nanomanifold import SO3

from body_models import config
from body_models.common import simplify_mesh
from body_models.cache import HF_MODEL_BASE_URL, download_and_extract, get_cache_dir

PathLike = Path | str

Front = tuple[list[int], list[int]]  # One FK depth level: (joint_indices, parent_indices).

__all__ = [
    "get_model_path",
    "download_model",
    "load_model_data",
    "MhrWeights",
    "compute_kinematic_fronts",
    "simplify_mesh",
    "load_pose_correctives_weights",
    "load_pose_correctives",
]

MHR_URL = f"{HF_MODEL_BASE_URL}/mhr/assets.zip"
SUPPORTED_LODS = tuple(range(7))


@dataclass(frozen=True)
class MhrWeights:
    base_vertices: np.ndarray
    blendshape_dirs: np.ndarray
    skin_weights: np.ndarray
    skin_indices: np.ndarray
    faces: np.ndarray
    joint_offsets: np.ndarray
    joint_pre_rotations: np.ndarray
    parameter_transform: np.ndarray
    bind_inv_linear: np.ndarray
    bind_inv_translation: np.ndarray
    corrective_W1: np.ndarray
    corrective_W2: np.ndarray
    parents: list[int]
    kinematic_fronts: list[Front]
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_file():
        raise ValueError(f"Expected an MHR model directory, got file: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"MHR model path {model_path} does not exist")
    model_file = model_path / "mhr_model.pt"
    if not model_file.is_file():
        raise FileNotFoundError(f"MHR model directory is missing mhr_model.pt: {model_path}")
    return model_path


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve MHR model path, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("mhr")

    if model_path is not None:
        return validate_path(model_path)

    cache_path = get_cache_dir() / "mhr"
    if _has_hosted_assets(cache_path):
        return cache_path

    return download_model()


def download_model() -> Path:
    """Download MHR model assets."""
    cache_dir = get_cache_dir() / "mhr"
    print(f"Downloading MHR model to {cache_dir}...")
    download_and_extract(url=MHR_URL, dest=cache_dir)
    print("Done")
    return cache_dir


def load_model_data(asset_dir: Path, *, lod: int = 1, simplify: float = 1.0) -> MhrWeights:
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    if lod not in SUPPORTED_LODS:
        raise ValueError(f"MHR lod must be one of {SUPPORTED_LODS}, got {lod}")

    shared_data = _load_raw_model_data(asset_dir)
    data = shared_data if lod == 1 else _load_preprocessed_lod_data(asset_dir, lod, shared_data)
    base_vertices = data["base_vertices"]
    blendshape_dirs = data["blendshape_dirs"]
    skin_weights = data["skin_weights"]
    skin_indices = data["skin_indices"].astype(np.int64)
    faces = data["faces"].astype(np.int64)
    corrective_weights = load_pose_correctives_weights(asset_dir, lod)
    corrective_W2 = corrective_weights["W2"]
    if corrective_W2.shape[0] != len(base_vertices) * 3:
        raise ValueError(
            f"MHR lod{lod} corrective W2 has {corrective_W2.shape[0]} rows, expected {len(base_vertices) * 3}"
        )

    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        base_vertices, faces, vertex_map = simplify_mesh(base_vertices, faces.astype(int), target_faces)
        blendshape_dirs = blendshape_dirs[:, vertex_map]
        skin_weights = skin_weights[vertex_map]
        skin_indices = skin_indices[vertex_map]
        corrective_W2_vertices = corrective_W2.reshape(-1, 3, corrective_W2.shape[-1])
        corrective_W2 = corrective_W2_vertices[vertex_map].reshape(-1, corrective_W2.shape[-1])

    inv_bind = data["inverse_bind_pose"]
    t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
    joint_parents = np.asarray(data["joint_parents"], dtype=np.int64)

    return MhrWeights(
        base_vertices=np.array(base_vertices, copy=True),
        blendshape_dirs=np.array(blendshape_dirs, copy=True),
        skin_weights=np.array(skin_weights, copy=True),
        skin_indices=np.array(skin_indices, copy=True),
        faces=np.array(faces, copy=True),
        joint_offsets=np.array(data["joint_offsets"], copy=True),
        joint_pre_rotations=np.array(data["joint_pre_rotations"], copy=True),
        parameter_transform=np.array(data["parameter_transform"], copy=True),
        bind_inv_linear=np.array(SO3.conversions.from_quat_to_rotmat(q, convention="xyzw") * s[..., None], copy=True),
        bind_inv_translation=np.array(t, copy=True),
        corrective_W1=np.array(corrective_weights["W1"], copy=True),
        corrective_W2=np.array(corrective_W2, copy=True),
        parents=joint_parents.tolist(),
        kinematic_fronts=compute_kinematic_fronts(joint_parents),
        joint_names=list(data["joint_names"]),
    )


def _load_raw_model_data(asset_dir: Path) -> dict[str, Any]:
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


def _load_preprocessed_lod_data(asset_dir: Path, lod: int, shared_data: dict[str, Any]) -> dict[str, Any]:
    path = asset_dir / f"mhr_lod{lod}.npz"
    with np.load(path, allow_pickle=False) as asset:
        joint_names = [str(name) for name in asset["skin_joint_names"].tolist()]
        checkpoint_joint_index = {name: index for index, name in enumerate(shared_data["joint_names"])}
        missing = sorted(name for name in joint_names if name not in checkpoint_joint_index)
        if missing:
            raise ValueError(f"{path} references joints missing from mhr_model.pt: {missing}")

        skin_joint_indices = np.asarray(asset["skin_joint_indices"], dtype=np.int64)
        mapped_joint_indices = np.asarray([checkpoint_joint_index[joint_names[index]] for index in skin_joint_indices])
        base_vertices = np.asarray(asset["base_vertices"], dtype=np.float32)
        skin_indices, skin_weights = _build_dense_skinning(
            asset["skin_vertex_indices"],
            mapped_joint_indices,
            asset["skin_weights"],
            len(base_vertices),
        )

        return shared_data | {
            "base_vertices": base_vertices,
            "blendshape_dirs": np.asarray(asset["blendshape_dirs"], dtype=np.float32),
            "skin_weights": skin_weights,
            "skin_indices": skin_indices,
            "faces": np.asarray(asset["faces"], dtype=np.int64),
        }


def _has_hosted_assets(model_path: Path) -> bool:
    asset_names = [
        "mhr_model.pt",
        "corrective_activation.npz",
        *(f"mhr_lod{lod}.npz" for lod in SUPPORTED_LODS),
    ]
    return all((model_path / name).is_file() for name in asset_names)


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
        from ptloader import load
    except ImportError as exc:
        raise ImportError("ptloader is required to load MHR checkpoints without torch.") from exc

    return load(checkpoint_path, weights_only=True)


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


def compute_kinematic_fronts(parents: np.ndarray) -> list[Front]:
    """Compute kinematic fronts for batched FK. Returns [(joint_indices, parent_indices), ...]."""
    parents_list = parents.tolist()

    n_joints = len(parents_list)
    processed: set[int] = set()
    fronts: list[Front] = []

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
            self._sparse_indices = nn.Buffer(idx, persistent=False)
            self.sparse_indices = nn.Parameter(idx, requires_grad=False)
            self.sparse_weight = nn.Parameter(torch.zeros(idx.shape[1]), requires_grad=False)
            self.dense_weight = nn.Buffer(torch.zeros(out_features, in_features), persistent=False)
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
            rot = SO3.conversions.from_euler_to_rotmat(euler, convention="xyz", xp=torch)
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
