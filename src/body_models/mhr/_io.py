"""I/O utilities for MHR model loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from ptloader import load as load_pytorch_checkpoint

from body_models import _config as config
from body_models._cache import download_hf_archive, get_cache_dir, write_npz_atomic
from body_models._common import Front, compute_kinematic_fronts, simplify_mesh, sparse
from body_models._common.skinning import CompactSkinning

PathLike = Path | str
SUPPORTED_LODS = tuple(range(7))
MHR_ASSETS = (
    "mhr_model.pt",
    "corrective_activation.npz",
    *(f"mhr_lod{lod}.npz" for lod in SUPPORTED_LODS),
    *(f"corrective_blendshapes_lod{lod}.npz" for lod in SUPPORTED_LODS),
)
_MHR_DEFAULT_ASSETS = (
    "mhr_model.pt",
    "corrective_activation.npz",
    "corrective_blendshapes_lod1.npz",
)

__all__ = [
    "MhrCorrectives",
    "MhrWeights",
    "compute_kinematic_fronts",
    "download_model",
    "get_model_path",
    "load_model_data",
    "load_pose_correctives_weights",
    "simplify_mesh",
]


@dataclass(frozen=True)
class MhrCorrectives:
    hidden_weights: Float[np.ndarray, "input hidden"]
    basis: sparse.SparseMatrix


@dataclass(frozen=True)
class MhrWeights:
    base_vertices: Float[np.ndarray, "V 3"]
    blendshape_dirs: Float[np.ndarray, "117 V 3"]
    compact_skinning: CompactSkinning
    dense_skin_weights: Float[np.ndarray, "V J"]
    faces: Int[np.ndarray, "F 3"]
    joint_offsets: Float[np.ndarray, "J 3"]
    joint_pre_rotations: Float[np.ndarray, "J 4"]
    parameter_transform: Float[np.ndarray, "D N"]
    bind_inv_linear: Float[np.ndarray, "J 3 3"]
    bind_inv_translation: Float[np.ndarray, "J 3"]
    correctives: MhrCorrectives
    parents: list[int]
    kinematic_fronts: list[Front]
    joint_names: list[str]


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_file():
        raise ValueError(f"Expected an MHR model directory, got file: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"MHR model path {model_path} does not exist")
    _require_assets(model_path, _MHR_DEFAULT_ASSETS)
    return model_path


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve MHR model path, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("mhr")

    if model_path is not None:
        return validate_path(model_path)

    cache_path = get_cache_dir() / "mhr"
    if _has_model(cache_path):
        return cache_path

    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download MHR model assets."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "mhr"
    if _has_model(output_dir):
        return validate_path(output_dir)
    print(f"Downloading MHR model to {output_dir}...")
    download_hf_archive("mhr/assets.zip", output_dir)
    print("Done")
    return validate_path(output_dir)


def load_model_data(asset_dir: Path, *, lod: int = 1, simplify: float = 1.0) -> MhrWeights:
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0")
    if lod not in SUPPORTED_LODS:
        raise ValueError(f"MHR lod must be one of {SUPPORTED_LODS}, got {lod}")

    asset_dir = validate_path(asset_dir)
    lod_assets = (f"corrective_blendshapes_lod{lod}.npz",)
    if lod != 1:
        lod_assets += (f"mhr_lod{lod}.npz",)
    _require_assets(asset_dir, lod_assets)

    data = _load_raw_model_data(asset_dir)
    if lod != 1:
        data = _load_lod_data(asset_dir, lod, data)
    base_vertices = data["base_vertices"]
    blendshape_dirs = data["blendshape_dirs"]
    skin_weights = data["skin_weights"]
    skin_indices = data["skin_indices"].astype(np.int32)
    faces = data["faces"].astype(np.int64)
    vertex_map = None
    if simplify > 1.0:
        target_faces = int(len(faces) / simplify)
        base_vertices, faces, vertex_map = simplify_mesh(base_vertices, faces.astype(int), target_faces)
        blendshape_dirs = blendshape_dirs[:, vertex_map]
        skin_weights = skin_weights[vertex_map]
        skin_indices = skin_indices[vertex_map]
    correctives = load_pose_correctives_weights(asset_dir, lod, vertex_map=vertex_map)

    inv_bind = data["inverse_bind_pose"]
    t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
    joint_parents = np.asarray(data["joint_parents"], dtype=np.int64)
    dense_skin_weights = _expand_skinning_weights(skin_indices, skin_weights, len(joint_parents))

    return MhrWeights(
        base_vertices=np.array(base_vertices, copy=True),
        blendshape_dirs=np.array(blendshape_dirs, copy=True),
        compact_skinning=CompactSkinning(
            np.array(skin_indices, copy=True),
            np.array(skin_weights, copy=True),
        ),
        dense_skin_weights=dense_skin_weights,
        faces=np.array(faces, copy=True),
        joint_offsets=np.array(data["joint_offsets"], copy=True),
        joint_pre_rotations=np.array(data["joint_pre_rotations"], copy=True),
        parameter_transform=np.array(data["parameter_transform"], copy=True),
        bind_inv_linear=np.array(SO3.conversions.from_quat_to_rotmat(q, convention="xyzw") * s[..., None], copy=True),
        bind_inv_translation=np.array(t, copy=True),
        correctives=correctives,
        parents=joint_parents.tolist(),
        kinematic_fronts=compute_kinematic_fronts(joint_parents),
        joint_names=list(data["joint_names"]),
    )


def _expand_skinning_weights(
    joint_indices: Int[np.ndarray, "V K"],
    joint_weights: Float[np.ndarray, "V K"],
    num_joints: int,
) -> Float[np.ndarray, "V J"]:
    num_vertices = joint_indices.shape[0]
    rows = np.broadcast_to(np.arange(num_vertices)[:, None], joint_indices.shape)
    valid = joint_indices >= 0
    dense = np.zeros((num_vertices, num_joints), dtype=joint_weights.dtype)
    np.add.at(dense, (rows[valid], joint_indices[valid]), joint_weights[valid])
    return dense


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


def _load_lod_data(asset_dir: Path, lod: int, data: dict[str, Any]) -> dict[str, Any]:
    path = asset_dir / f"mhr_lod{lod}.npz"
    with np.load(path, allow_pickle=False) as asset:
        joint_names = [str(name) for name in asset["skin_joint_names"].tolist()]
        checkpoint_joint_index = {name: index for index, name in enumerate(data["joint_names"])}
        skin_joint_indices = np.asarray(asset["skin_joint_indices"], dtype=np.int64)
        mapped_joint_indices = np.asarray([checkpoint_joint_index[name] for name in joint_names], dtype=np.int64)
        base_vertices = np.asarray(asset["base_vertices"], dtype=np.float32)
        skin_indices, skin_weights = _build_dense_skinning(
            asset["skin_vertex_indices"],
            mapped_joint_indices[skin_joint_indices],
            asset["skin_weights"],
            len(base_vertices),
        )
        blendshape_dirs = np.asarray(asset["blendshape_dirs"], dtype=np.float32)
        faces = np.asarray(asset["faces"], dtype=np.int64)

    return data | {
        "base_vertices": base_vertices,
        "blendshape_dirs": blendshape_dirs,
        "skin_weights": skin_weights,
        "skin_indices": skin_indices,
        "faces": faces,
    }


def _has_model(model_path: Path) -> bool:
    return all((model_path / name).is_file() for name in MHR_ASSETS)


def _get_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def _load_checkpoint_numpy(checkpoint_path: Path) -> Any:
    return load_pytorch_checkpoint(checkpoint_path, weights_only=True)


def _build_dense_skinning(
    vert_indices: Int[np.ndarray, "N"],
    joint_indices: Int[np.ndarray, "N"],
    joint_weights: Float[np.ndarray, "N"],
    num_vertices: int,
) -> tuple[Int[np.ndarray, "V K"], Float[np.ndarray, "V K"]]:
    """Build dense skinning matrices from sparse representation."""
    vert_indices = vert_indices.astype(np.int64, copy=False)
    joint_indices = joint_indices.astype(np.int32, copy=False)
    counts = np.bincount(vert_indices, minlength=num_vertices)
    K = int(counts.max())

    dense_indices = np.zeros((num_vertices, K), dtype=np.int32)
    dense_weights = np.zeros((num_vertices, K), dtype=joint_weights.dtype)

    offsets = np.zeros(num_vertices + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    for v in range(num_vertices):
        start, end = int(offsets[v]), int(offsets[v + 1])
        dense_indices[v, : end - start] = joint_indices[start:end]
        dense_weights[v, : end - start] = joint_weights[start:end]

    return dense_indices, dense_weights


def load_pose_correctives_weights(
    asset_dir: Path,
    lod: int,
    *,
    vertex_map: Int[np.ndarray, "V"] | None = None,
) -> MhrCorrectives:
    """Load MHR pose-corrective weights in sparse input-output orientation."""
    with np.load(asset_dir / "corrective_activation.npz") as data:
        sparse_indices = data["0.sparse_indices"]
        sparse_values = data["0.sparse_weight"]
    out_features, in_features = 125 * 24, 125 * 6
    hidden_weights = np.zeros((in_features, out_features), dtype=np.float32)
    hidden_weights[sparse_indices[1], sparse_indices[0]] = sparse_values

    output_weights = _load_output_weights(asset_dir, lod)
    if vertex_map is not None:
        selected_columns = (vertex_map[:, None] * 3 + np.arange(3)).reshape(-1)
        output_weights = sparse.select_columns(output_weights, selected_columns)
    output_weights = sparse.scaled(output_weights, 0.01)

    return MhrCorrectives(
        hidden_weights=hidden_weights,
        basis=output_weights,
    )


def _load_output_weights(asset_dir: Path, lod: int) -> sparse.SparseMatrix:
    cache_file = _output_weights_cache_file(asset_dir, lod)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            shape = data["shape"]
            return sparse.SparseMatrix(
                row_indices=np.asarray(data["rows"], dtype=np.int64).copy(),
                column_indices=np.asarray(data["columns"], dtype=np.int64).copy(),
                values=np.asarray(data["values"], dtype=np.float32).copy(),
                shape=(int(shape[0]), int(shape[1])),
            )

    with np.load(asset_dir / f"corrective_blendshapes_lod{lod}.npz") as data:
        blendshapes = np.asarray(data["corrective_blendshapes"], dtype=np.float32)
    num_components = blendshapes.shape[0]
    output_weights = sparse.from_dense(blendshapes.reshape(num_components, -1))
    write_npz_atomic(
        cache_file,
        rows=output_weights.row_indices,
        columns=output_weights.column_indices,
        values=output_weights.values,
        shape=np.asarray(output_weights.shape, dtype=np.int64),
    )
    return output_weights


def _output_weights_cache_file(asset_dir: Path, lod: int) -> Path:
    source = asset_dir / f"corrective_blendshapes_lod{lod}.npz"
    stat = source.stat()
    key = hashlib.md5(f"v1:{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    cache_dir = get_cache_dir() / "mhr" / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"correctives_{key}.npz"


def _require_assets(model_path: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MHR model directory is missing required assets: {', '.join(missing)}")
