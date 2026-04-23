"""I/O utilities for SOMA model loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse
from scipy.sparse import csc_matrix
from jaxtyping import Float, Int

from .. import config
from ..common import simplify_mesh
from ..utils import get_cache_dir

Front = tuple[list[int], list[int]]

SOMA_CORE_ASSET = "SOMA_neutral.npz"
SOMA_CORRECTIVES_ASSET = "correctives_model.pt"
SOMA_ASSETS = (SOMA_CORE_ASSET, SOMA_CORRECTIVES_ASSET)
SOMA_BASE_URL = "https://huggingface.co/nvidia/SOMA-X/resolve/main"

__all__ = [
    "get_model_path",
    "download_model",
    "load_model_data",
    "get_identity_model_path",
    "load_identity_transfer_data",
    "load_pose_correctives_weights",
    "compute_kinematic_fronts",
    "simplify_mesh",
]


@dataclass(frozen=True)
class _SparseCoo:
    indices: Int[np.ndarray, "D NNZ"]
    values: Float[np.ndarray, "NNZ"]
    size: tuple[int, ...]
    is_coalesced: bool


@dataclass(frozen=True)
class _ModelTypeSpec:
    identity_dim: int
    num_scale_params: int | None = None
    default_identity_value: float = 0.0
    source_scale: float = 1.0
    output_scale: float = 1.0
    config_key: str | None = None
    asset_dir: str | None = None
    source_mesh_name: str | None = None
    target_mesh_name: str | None = None
    requires_direct_file: bool = False
    filename_hint: str | None = None
    use_laplacian: bool = True


MODEL_TYPE_SPECS = {
    "soma": _ModelTypeSpec(identity_dim=128),
    "mhr": _ModelTypeSpec(
        identity_dim=45,
        num_scale_params=68,
        source_scale=100.0,
        config_key="mhr",
        asset_dir="MHR",
        source_mesh_name="base_body_lod1.obj",
        target_mesh_name="SOMA_wrap_lod1.obj",
    ),
    "anny": _ModelTypeSpec(
        identity_dim=6,
        default_identity_value=0.5,
        output_scale=100.0,
        config_key="anny",
        asset_dir="Anny",
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
        use_laplacian=False,
    ),
    "smpl": _ModelTypeSpec(
        identity_dim=10,
        output_scale=100.0,
        config_key="smpl-neutral",
        asset_dir="SMPL",
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
        requires_direct_file=True,
        filename_hint="SMPL_NEUTRAL",
    ),
    "smplx": _ModelTypeSpec(
        identity_dim=10,
        output_scale=100.0,
        config_key="smplx-neutral",
        asset_dir="SMPLX",
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
        requires_direct_file=True,
        filename_hint="SMPLX_NEUTRAL",
    ),
}
IDENTITY_MODEL_TYPES = tuple(name for name, spec in MODEL_TYPE_SPECS.items() if spec.asset_dir is not None)


def get_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve SOMA model directory, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("soma")

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.is_file():
            raise ValueError(
                f"Expected a SOMA asset directory, got file: {model_path}\n"
                f"Please provide a directory containing {SOMA_CORE_ASSET}."
            )
        if model_path.is_dir():
            missing = _missing_assets(model_path)
            if not missing:
                return model_path
            if (model_path / SOMA_CORE_ASSET).exists():
                return download_model(model_path)
            raise FileNotFoundError(f"SOMA model path {model_path} is missing required assets: {', '.join(missing)}.")
        raise FileNotFoundError(f"SOMA model path {model_path} does not exist.")

    cache_path = get_cache_dir() / "soma"
    if not _missing_assets(cache_path):
        return cache_path

    return download_model()


def download_model(model_dir: Path | str | None = None) -> Path:
    """Download SOMA assets from Hugging Face."""
    import urllib.request

    cache_dir = Path(model_dir) if model_dir is not None else get_cache_dir() / "soma"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = _missing_assets(cache_dir)
    if missing:
        print(f"Downloading SOMA model to {cache_dir}...")
        for name in missing:
            urllib.request.urlretrieve(f"{SOMA_BASE_URL}/{name}", cache_dir / name)
        print("Done")
    return cache_dir


def ensure_identity_assets(model_dir: Path, model_type: str) -> None:
    """Ensure supplementary SOMA assets exist for a given identity backend."""
    import urllib.request

    normalized = model_type.lower()
    spec = MODEL_TYPE_SPECS.get(normalized)
    if spec is None or spec.asset_dir is None:
        raise ValueError(f"Unsupported SOMA identity assets: {model_type}")

    asset_dir = Path(model_dir)
    asset_names = (
        f"{spec.asset_dir}/{spec.target_mesh_name}",
        f"{spec.asset_dir}/{spec.source_mesh_name}",
    )
    missing = [name for name in asset_names if not (asset_dir / name).exists()]
    if missing:
        print(f"Downloading SOMA {normalized} assets to {asset_dir}...")
        for name in missing:
            path = asset_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{SOMA_BASE_URL}/{name}", path)
        print("Done")


def get_identity_model_path(model_type: str) -> Path | None:
    normalized = model_type.lower()
    spec = MODEL_TYPE_SPECS.get(normalized)
    if spec is None or spec.config_key is None:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}")

    model_path = config.get_model_path(spec.config_key)
    if model_path is None:
        return None

    path = Path(model_path)
    if spec.requires_direct_file and path.is_dir():
        raise ValueError(
            f"Directory paths are no longer supported for {normalized}: {path}\n"
            f"Please set {spec.config_key} to a direct {spec.filename_hint}.npz or {spec.filename_hint}.pkl path."
        )
    return path


def compute_kinematic_fronts(parents: np.ndarray | list[int]) -> list[Front]:
    """Compute kinematic fronts for batched FK."""
    parents_list = parents.tolist() if isinstance(parents, np.ndarray) else list(parents)

    n_joints = len(parents_list)
    processed: set[int] = set()
    fronts: list[Front] = []

    while len(processed) < n_joints:
        joints: list[int] = []
        joint_parents: list[int] = []
        for j, parent in enumerate(parents_list):
            if j in processed:
                continue
            if parent < 0 or parent == j or parent in processed:
                joints.append(j)
                joint_parents.append(-1 if parent == j else int(parent))
        if not joints:
            raise ValueError(f"Invalid SOMA parent chain: {parents_list}")
        fronts.append((joints, joint_parents))
        processed.update(joints)

    return fronts


def _missing_assets(model_dir: Path) -> list[str]:
    return [name for name in SOMA_ASSETS if not (model_dir / name).exists()]


def _identity_transfer_cache_file(asset_dir: Path, model_type: str) -> Path:
    preprocessed_dir = get_cache_dir() / "soma" / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"identity-transfer-v2:{model_type}:{asset_dir.resolve()}".encode()).hexdigest()
    return preprocessed_dir / f"identity_transfer_{key}.npz"


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(f"SOMA identity backends require trimesh to load {path.name}.") from exc

    mesh = cast(Any, trimesh.load(path, maintain_order=True, process=False))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def _fabricate_tet(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return p0 + np.cross(p1 - p0, p2 - p0, axis=-1)


def _compute_barycentric_coords_3d(
    p: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
    rhs = p - v0
    b123 = np.linalg.solve(T, rhs[..., None]).squeeze(-1)
    b0 = 1.0 - b123.sum(axis=-1, keepdims=True)
    return np.concatenate([b0, b123], axis=-1).astype(np.float32, copy=False)


def _compute_identity_correspondence(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("SOMA identity backends require trimesh to precompute topology transfer.") from exc

    mesh = trimesh.Trimesh(vertices=source_vertices, faces=source_faces, process=False)
    _closest_points, _distance, face_ids = mesh.nearest.on_surface(target_vertices)
    face_ids = np.asarray(face_ids, dtype=np.int64)

    fabricated = _fabricate_tet(
        source_vertices[source_faces[:, 0]],
        source_vertices[source_faces[:, 1]],
        source_vertices[source_faces[:, 2]],
    )
    source_tetrahedra = np.concatenate(
        [source_faces, np.arange(len(source_faces), dtype=np.int64)[:, None] + len(source_vertices)],
        axis=1,
    )
    source_vertices_tet = np.concatenate([source_vertices, fabricated], axis=0)
    tet_indices = source_tetrahedra[face_ids]
    bary_coords = _compute_barycentric_coords_3d(
        target_vertices,
        source_vertices_tet[tet_indices[:, 0]],
        source_vertices_tet[tet_indices[:, 1]],
        source_vertices_tet[tet_indices[:, 2]],
        source_vertices_tet[tet_indices[:, 3]],
    )
    return source_tetrahedra, face_ids, bary_coords


def _build_cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> scipy_sparse.csr_matrix:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    def _cotangent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a * b, axis=-1)
        cross = np.cross(a, b, axis=-1)
        return dot / (np.linalg.norm(cross, axis=-1) + 1e-8)

    cot0 = _cotangent(e1, e2)
    cot1 = _cotangent(e2, e0)
    cot2 = _cotangent(e0, e1)

    row_ids = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
    col_ids = np.concatenate([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
    values = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]).astype(np.float32, copy=False)

    num_vertices = len(vertices)
    weights = scipy_sparse.coo_matrix((values, (row_ids, col_ids)), shape=(num_vertices, num_vertices)).tocsr()
    weights = ((weights + weights.T) * 0.5).tocsr()
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    return (scipy_sparse.diags(row_sums) - weights).tocsr()


def _build_identity_laplacian_data(
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    unknown_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    laplacian = _build_cotangent_laplacian(target_vertices, target_faces)
    unknown_ids = np.asarray(np.unique(unknown_ids), dtype=np.int64)
    anchor_mask = np.ones(len(target_vertices), dtype=bool)
    anchor_mask[unknown_ids] = False
    anchor_ids = np.flatnonzero(anchor_mask).astype(np.int64)

    lap_u = laplacian[unknown_ids]
    solve_matrix = (-laplacian[unknown_ids][:, unknown_ids].toarray()).astype(np.float32, copy=False)
    anchor_matrix = (-laplacian[unknown_ids][:, anchor_ids].toarray()).astype(np.float32, copy=False)
    rhs_base = (-(lap_u @ target_vertices)).astype(np.float32, copy=False)
    return unknown_ids, anchor_ids, solve_matrix, anchor_matrix, rhs_base


def load_identity_transfer_data(asset_dir: Path, model_type: str) -> dict[str, np.ndarray]:
    cache_file = _identity_transfer_cache_file(asset_dir, model_type)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            return {name: np.asarray(data[name]).copy() for name in data.files}

    normalized = model_type.lower()
    spec = MODEL_TYPE_SPECS.get(normalized)
    if spec is None or spec.asset_dir is None or spec.source_mesh_name is None or spec.target_mesh_name is None:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}")

    ensure_identity_assets(asset_dir, normalized)
    mesh_dir = asset_dir / spec.asset_dir
    source_vertices, source_faces = _load_mesh(mesh_dir / spec.source_mesh_name)
    target_vertices, target_faces = _load_mesh(mesh_dir / spec.target_mesh_name)
    source_tetrahedra, face_ids, bary_coords = _compute_identity_correspondence(
        source_vertices=source_vertices,
        source_faces=source_faces,
        target_vertices=target_vertices,
    )

    if not spec.use_laplacian:
        unknown_ids = np.empty((0,), dtype=np.int64)
        anchor_ids = np.empty((0,), dtype=np.int64)
        solve_matrix = np.empty((0, 0), dtype=np.float32)
        anchor_matrix = np.empty((0, 0), dtype=np.float32)
        rhs_base = np.empty((0, 3), dtype=np.float32)
    else:
        facial_inner_vertices = load_model_data(asset_dir)["facial_inner_vertices"]
        unknown_ids, anchor_ids, solve_matrix, anchor_matrix, rhs_base = _build_identity_laplacian_data(
            target_vertices=target_vertices,
            target_faces=target_faces,
            unknown_ids=facial_inner_vertices,
        )

    np.savez_compressed(
        cache_file,
        source_vertices=source_vertices,
        source_tetrahedra=source_tetrahedra,
        face_ids=face_ids,
        bary_coords=bary_coords,
        unknown_ids=unknown_ids,
        anchor_ids=anchor_ids,
        solve_matrix=solve_matrix,
        anchor_matrix=anchor_matrix,
        rhs_base=rhs_base,
    )
    return {
        "source_vertices": source_vertices,
        "source_tetrahedra": source_tetrahedra,
        "face_ids": face_ids,
        "bary_coords": bary_coords,
        "unknown_ids": unknown_ids,
        "anchor_ids": anchor_ids,
        "solve_matrix": solve_matrix,
        "anchor_matrix": anchor_matrix,
        "rhs_base": rhs_base,
    }


def _correctives_cache_file(asset_dir: Path) -> Path:
    preprocessed_dir = get_cache_dir() / "soma" / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"v3:{(asset_dir / SOMA_CORRECTIVES_ASSET).resolve()}".encode()).hexdigest()
    return preprocessed_dir / f"correctives_{key}.npz"


def _get_layout(name: str) -> str:
    return name


def _rebuild_sparse_tensor(layout: str, payload: tuple[Any, Any, tuple[int, ...], bool]) -> _SparseCoo:
    if layout != "torch.sparse_coo":
        raise ValueError(f"Unsupported SOMA sparse layout: {layout}")
    indices_ref, values_ref, size, is_coalesced = payload
    return _SparseCoo(
        indices=indices_ref.to_numpy().astype(np.int64, copy=False),
        values=values_ref.to_numpy().astype(np.float32, copy=False),
        size=tuple(int(v) for v in size),
        is_coalesced=bool(is_coalesced),
    )


def _load_sparse_checkpoint_numpy(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SOMA corrective checkpoint not found: {checkpoint_path}")

    try:
        from ptloader import load
    except ImportError as exc:
        raise ImportError("ptloader is required to load SOMA corrective checkpoints.") from exc

    return load(
        checkpoint_path,
        weights_only=True,
        pickle_global_registry={
            ("torch.serialization", "_get_layout"): _get_layout,
            ("torch._utils", "_rebuild_sparse_tensor"): _rebuild_sparse_tensor,
            ("torch", "Size"): tuple,
        },
    )


def _as_dense_float32(value: np.ndarray | _SparseCoo) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    return _dense_from_sparse(value)


def _dense_from_sparse(sparse: _SparseCoo) -> np.ndarray:
    dense = np.zeros(sparse.size, dtype=np.float32)
    dense[tuple(sparse.indices)] = sparse.values
    return dense


def load_pose_correctives_weights(asset_dir: Path) -> dict[str, Any]:
    """Load SOMA pose-corrective weights in backend-agnostic form."""
    cache_file = _correctives_cache_file(asset_dir)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            return {
                "bindpose": np.asarray(data["bindpose"], dtype=np.float32).copy(),
                "W1": np.asarray(data["W1"], dtype=np.float32).copy(),
                "W2_rows": np.asarray(data["W2_rows"], dtype=np.int64).copy(),
                "W2_cols": np.asarray(data["W2_cols"], dtype=np.int64).copy(),
                "W2_values": np.asarray(data["W2_values"], dtype=np.float32).copy(),
                "num_vertices": int(data["num_vertices"][0]),
                "use_tanh": bool(data["use_tanh"][0]),
            }

    checkpoint_path = asset_dir / SOMA_CORRECTIVES_ASSET
    ckpt = _load_sparse_checkpoint_numpy(checkpoint_path)

    W1_sparse = cast(_SparseCoo, ckpt["W1"])
    W2_sparse = cast(_SparseCoo, ckpt["W2"])
    bindpose = np.asarray(cast(np.ndarray, ckpt["bindpose"]), dtype=np.float32)
    cors_per_joint = int(ckpt["C_max"])
    W1 = _dense_from_sparse(W1_sparse)
    W2_rows = W2_sparse.indices[0].astype(np.int64, copy=False)
    W2_cols = W2_sparse.indices[1].astype(np.int64, copy=False)
    W2_values = W2_sparse.values.astype(np.float32, copy=False)

    if "M1_mask" in ckpt:
        M1_mask = _as_dense_float32(cast(np.ndarray | _SparseCoo, ckpt["M1_mask"]))
        W1 *= np.repeat(np.repeat(M1_mask, 6, axis=0), cors_per_joint, axis=1)

    if "M2_mask" in ckpt:
        M2_mask = _as_dense_float32(cast(np.ndarray | _SparseCoo, ckpt["M2_mask"]))
        scale = M2_mask[W2_rows // cors_per_joint, W2_cols // 3].astype(np.float32, copy=False)
        keep = scale != 0.0
        W2_rows = W2_rows[keep]
        W2_cols = W2_cols[keep]
        W2_values = W2_values[keep] * scale[keep]

    num_vertices = W2_sparse.size[1] // 3
    use_tanh = bool(ckpt["use_tanh"])

    np.savez_compressed(
        cache_file,
        bindpose=bindpose,
        W1=W1,
        W2_rows=W2_rows,
        W2_cols=W2_cols,
        W2_values=W2_values,
        num_vertices=np.array([num_vertices], dtype=np.int64),
        use_tanh=np.array([use_tanh], dtype=np.bool_),
    )

    return {
        "bindpose": bindpose.copy(),
        "W1": W1.copy(),
        "W2_rows": W2_rows.copy(),
        "W2_cols": W2_cols.copy(),
        "W2_values": W2_values.copy(),
        "num_vertices": num_vertices,
        "use_tanh": use_tanh,
    }


def _get_joint_children_ids(parents: np.ndarray) -> list[list[int]]:
    parent_ids = parents.tolist()
    children = [[] for _ in range(len(parent_ids))]
    for i in range(1, len(parent_ids)):
        children[parent_ids[i]].append(i)
    return children


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _linear_rbf(r: np.ndarray) -> np.ndarray:
    return r


def _get_basis_weights(control_points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    """Compute dense linear-RBF interpolation weights for one query point."""
    num_points, dim = control_points.shape

    K = _linear_rbf(_pairwise_dist(control_points, control_points)).astype(np.float64, copy=False)
    K[np.diag_indices(num_points)] += 1e-8

    ones = np.ones((num_points, 1), dtype=np.float64)
    P = np.concatenate([ones, control_points.astype(np.float64, copy=False)], axis=1)
    Z = np.zeros((dim + 1, dim + 1), dtype=np.float64)
    A = np.block([[K, P], [P.T, Z]])

    rhs = np.concatenate(
        [
            _linear_rbf(np.linalg.norm(control_points - query_point[None, :], axis=1)).astype(np.float64, copy=False),
            np.array([1.0], dtype=np.float64),
            query_point.astype(np.float64, copy=False),
        ]
    )
    lu, piv = scipy_linalg.lu_factor(A)
    weights = scipy_linalg.lu_solve((lu, piv), rhs)
    return weights[:num_points].astype(np.float32, copy=False)


def _build_joint_position_regressor(
    bind_shape: np.ndarray,
    bind_world_transforms: np.ndarray,
    skin_weights: np.ndarray,
    joint_parents: np.ndarray,
    vertex_ids_to_exclude: np.ndarray | None,
) -> np.ndarray:
    """Precompute dense vertex-to-joint regressors used by SOMA skeleton fitting."""
    regressor_mask = (skin_weights > 0.0) & (skin_weights[:, joint_parents] > 0.0)
    zero_weight_ids = np.where(regressor_mask.sum(axis=0) == 0.0)[0]

    joint_parents_cur = joint_parents.copy()
    if len(zero_weight_ids) > 0:
        regressor_mask[:, zero_weight_ids] = skin_weights[:, zero_weight_ids] > 0.0

    while len(zero_weight_ids) > 1:
        parent_cols = joint_parents_cur[zero_weight_ids]
        regressor_mask[:, zero_weight_ids] |= skin_weights[:, parent_cols] > 0.0
        zero_weight_ids = np.where(regressor_mask.sum(axis=0) == 0.0)[0]
        next_parents = joint_parents[joint_parents_cur]
        if np.array_equal(next_parents, joint_parents_cur):
            break
        joint_parents_cur = next_parents

    if np.array_equal(zero_weight_ids, np.array([0, 1], dtype=np.int64)):
        child_ids = _get_joint_children_ids(joint_parents)[1]
        regressor_mask[:, 1] = regressor_mask[:, child_ids].any(axis=1)

    if vertex_ids_to_exclude is not None and len(vertex_ids_to_exclude) > 0:
        regressor_mask[np.asarray(vertex_ids_to_exclude, dtype=np.int64)] = False

    num_joints = bind_world_transforms.shape[0]
    num_vertices = bind_shape.shape[0]
    joint_regressor = np.zeros((num_joints, num_vertices), dtype=np.float32)

    for joint_index in range(1, num_joints):
        control_mask = regressor_mask[:, joint_index]
        if not np.any(control_mask):
            continue
        control_points = bind_shape[control_mask]
        query_point = bind_world_transforms[joint_index, :3, 3]
        joint_regressor[joint_index, np.where(control_mask)[0]] = _get_basis_weights(control_points, query_point)

    return joint_regressor


@lru_cache(maxsize=None)
def _load_model_data_cached(model_dir: str) -> dict[str, Any]:
    asset_dir = Path(model_dir)
    with np.load(asset_dir / SOMA_CORE_ASSET, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        num_vertices = mean.shape[0]
        shapedirs = np.asarray(data["shapedirs"], dtype=np.float32).reshape(-1, num_vertices, 3)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float32)
        faces = np.asarray(data["triangles"], dtype=np.int64)
        bind_shape = np.asarray(data["bind_shape"], dtype=np.float32)
        bind_pose_world = np.asarray(data["bind_pose_world"], dtype=np.float32)
        bind_pose_local = np.asarray(data["bind_pose_local"], dtype=np.float32)
        t_pose_world = np.asarray(data["t_pose_world"], dtype=np.float32)
        t_pose_local = np.asarray(data["t_pose_local"], dtype=np.float32)
        joint_parents_full = np.asarray(data["joint_parent_ids"], dtype=np.int64)
        joint_names_full = [str(name) for name in data["joint_names"]]

        skin_weights = csc_matrix(
            (
                data["skinning_weights_data"],
                data["skinning_weights_indices"],
                data["skinning_weights_indptr"],
            ),
            shape=tuple(int(x) for x in data["skinning_weights_shape"]),
        ).toarray()
        skin_weights = np.asarray(skin_weights, dtype=np.float32)

        facial_inner = np.concatenate(
            [
                np.asarray(data["segment_eye_bags"], dtype=np.int64),
                np.asarray(data["segment_mouth_bag"], dtype=np.int64),
            ]
        )
    joint_regressor = _build_joint_position_regressor(
        bind_shape=bind_shape,
        bind_world_transforms=bind_pose_world,
        skin_weights=skin_weights,
        joint_parents=joint_parents_full,
        vertex_ids_to_exclude=facial_inner,
    )

    public_parents = (joint_parents_full[1:] - 1).astype(np.int64).tolist()
    joint_children_full = _get_joint_children_ids(joint_parents_full)
    skinned_vertex_indices_full = [
        np.where(skin_weights[:, joint_index] > 0.01)[0].astype(np.int64).tolist()
        for joint_index in range(skin_weights.shape[1])
    ]

    return {
        "mean": mean,
        "shapedirs": shapedirs,
        "eigenvalues": eigenvalues,
        "faces": faces,
        "bind_shape": bind_shape,
        "bind_pose_world": bind_pose_world,
        "bind_pose_local": bind_pose_local,
        "t_pose_world": t_pose_world,
        "t_pose_local": t_pose_local,
        "joint_parents_full": joint_parents_full,
        "joint_names_full": joint_names_full,
        "joint_regressor": joint_regressor,
        "skin_weights_full": skin_weights,
        "facial_inner_vertices": facial_inner,
        "skinned_vertex_indices_full": skinned_vertex_indices_full,
        "joint_children_full": joint_children_full,
        "kinematic_fronts_full": compute_kinematic_fronts(joint_parents_full),
        "joint_names": joint_names_full[1:],
        "parents": public_parents,
    }


def load_model_data(model_path: Path) -> dict[str, Any]:
    """Load SOMA model data from disk."""
    return _load_model_data_cached(str(Path(model_path).resolve()))
