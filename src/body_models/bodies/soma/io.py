"""I/O utilities for SOMA model loading."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse
from scipy.sparse import csc_matrix
from jaxtyping import Float, Int

from body_models import config
from body_models.common import simplify_mesh
from body_models.cache import HF_DATASET_BASE_URL, download_file, get_cache_dir

PathLike = Path | str

Front = tuple[list[int], list[int]]

SOMA_CORE_ASSET = "SOMA_neutral.npz"
SOMA_CORRECTIVES_ASSET = "correctives_model.pt"
SOMA_TEMPLATE_RIG_ASSET = "SOMA_template_rig.usda"
SOMA_PROCEDURAL_TRANSFORMS_ASSET = "SOMA_procedural_transforms.json"
SOMA_ASSETS = (SOMA_CORE_ASSET, SOMA_CORRECTIVES_ASSET)
SOMA_UPSTREAM_02_ASSETS = (SOMA_TEMPLATE_RIG_ASSET, SOMA_PROCEDURAL_TRANSFORMS_ASSET)
SOMA_BASE_URL = f"{HF_DATASET_BASE_URL}/soma"
SOMA_LEGACY_NPZ_FIELDS = (
    "bind_shape",
    "bind_pose_world",
    "bind_pose_local",
    "t_pose_world",
    "t_pose_local",
    "joint_parent_ids",
    "joint_names",
    "skinning_weights_data",
    "skinning_weights_indices",
    "skinning_weights_indptr",
    "skinning_weights_shape",
)
SOMA_PROCEDURAL_NPZ_FIELDS = (
    "procedural_public_joint_indices_full",
    "procedural_rotation_matrix",
    "procedural_translation_matrix",
    "procedural_source_axis_ids",
    "procedural_source_axis_signs",
    "procedural_twist_joint_indices",
    "procedural_twist_axis_ids",
    "procedural_twist_axis_signs",
)

__all__ = [
    "SomaIdentityTransfer",
    "SomaPublicRig",
    "SomaProceduralRig",
    "SomaWeights",
    "get_model_path",
    "download_model",
    "load_model_data",
    "get_identity_model_path",
    "load_identity_transfer_data",
    "preprocess_model",
    "compute_kinematic_fronts",
    "active_public_skin_weights",
    "public_joint_metadata",
    "simplify_mesh",
]


@dataclass(frozen=True)
class _SparseCoo:
    indices: Int[np.ndarray, "D NNZ"]
    values: Float[np.ndarray, "NNZ"]
    size: tuple[int, ...]
    is_coalesced: bool


@dataclass(frozen=True)
class SomaCorrectives:
    corrective_bindpose: Float[np.ndarray, "Jf 3 3"]
    corrective_W1: Float[np.ndarray, "D K"]
    corrective_W2_rows: Int[np.ndarray, "NNZ"]
    corrective_W2_cols: Int[np.ndarray, "NNZ"]
    corrective_W2_values: Float[np.ndarray, "NNZ"]


@dataclass(frozen=True)
class SomaTopology:
    parents_full: list[int]
    joint_children_full: list[list[int]]
    joint_children_indices_full: Int[np.ndarray, "Jf C"]
    skinned_vertex_indices_full: list[list[int]]
    skinned_vertex_indices_full_index: Int[np.ndarray, "Jf K"]
    kinematic_fronts_full: list[Front]


@dataclass(frozen=True)
class SomaProceduralRig:
    public_joint_indices_full: Int[np.ndarray, "Jp"]
    rotation_matrix: Float[np.ndarray, "T Jp"]
    translation_matrix: Float[np.ndarray, "Jf Jf"]
    source_axis_ids: Int[np.ndarray, "Jp"]
    source_axis_signs: Float[np.ndarray, "Jp"]
    twist_joint_indices: Int[np.ndarray, "T"]
    twist_axis_ids: Int[np.ndarray, "T"]
    twist_axis_signs: Float[np.ndarray, "T"]


@dataclass(frozen=True)
class SomaPublicRig:
    joint_names_full: list[str]
    parents_full: list[int]
    bind_pose_world: Float[np.ndarray, "Jp 4 4"]
    bind_pose_local: Float[np.ndarray, "Jp 4 4"]
    t_pose_world: Float[np.ndarray, "Jp 4 4"]
    joint_regressor: Float[np.ndarray, "Jp Vf"]
    skin_weights_full: Float[np.ndarray, "Vf Jp"]
    skin_weights_active: Float[np.ndarray, "Va Jp"]
    topology: SomaTopology
    procedural: SomaProceduralRig


@dataclass(frozen=True)
class SomaWeights:
    mean_full: Float[np.ndarray, "Vf 3"]
    mean_active: Float[np.ndarray, "Va 3"]
    shapedirs_full: Float[np.ndarray, "S Vf 3"]
    shapedirs_active: Float[np.ndarray, "S Va 3"]
    eigenvalues: Float[np.ndarray, "S"]
    bind_shape_full: Float[np.ndarray, "Vf 3"]
    bind_pose_world: Float[np.ndarray, "Jf 4 4"]
    bind_pose_local: Float[np.ndarray, "Jf 4 4"]
    t_pose_world: Float[np.ndarray, "Jf 4 4"]
    t_pose_local: Float[np.ndarray, "Jf 4 4"]
    joint_regressor: Float[np.ndarray, "Jf Vf"]
    skin_weights_full: Float[np.ndarray, "Vf Jf"]
    skin_weights_active: Float[np.ndarray, "Va Jf"]
    skin_joint_indices_active: Int[np.ndarray, "Va K"]
    skin_joint_weights_active: Float[np.ndarray, "Va K"]
    faces: Int[np.ndarray, "F 3"]
    vertex_map: Int[np.ndarray, "Va"] | None
    facial_inner_vertices: Int[np.ndarray, "Va"]
    topology: SomaTopology
    correctives: SomaCorrectives
    joint_names_full: list[str]
    public: SomaPublicRig | None = None


@dataclass(frozen=True)
class SomaIdentityTransfer:
    source_vertices: Float[np.ndarray, "Vs 3"]
    source_tetrahedra: Int[np.ndarray, "Fs 4"]
    face_ids: Int[np.ndarray, "Vt"]
    bary_coords: Float[np.ndarray, "Vt 4"]
    unknown_ids: Int[np.ndarray, "U"]
    anchor_ids: Int[np.ndarray, "A"]
    solve_matrix: Float[np.ndarray, "U U"]
    anchor_matrix: Float[np.ndarray, "U A"]
    rhs_base: Float[np.ndarray, "U 3"]
    internal_to_source_rotation: Float[np.ndarray, "3 3"]
    internal_to_source_translation: Float[np.ndarray, "3"]
    source_to_soma_rotation: Float[np.ndarray, "3 3"]
    source_scale: float
    output_scale: float


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


def validate_path(model_path: PathLike) -> Path:
    model_path = Path(model_path)
    if model_path.is_file():
        raise ValueError(f"Expected a SOMA asset directory, got file: {model_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"SOMA model path {model_path} does not exist.")
    missing = _missing_assets(model_path)
    if missing:
        raise FileNotFoundError(f"SOMA model path {model_path} is missing required assets: {', '.join(missing)}.")
    unsupported = _missing_legacy_npz_fields(model_path)
    if unsupported:
        missing_sidecars = [name for name in SOMA_UPSTREAM_02_ASSETS if not (model_path / name).exists()]
        if not missing_sidecars:
            raise FileNotFoundError(_preprocess_required_message(model_path, unsupported))
        raise FileNotFoundError(
            f"SOMA model path {model_path} is missing required NPZ fields: {', '.join(unsupported)}. "
            f"Missing upstream 0.2.1 sidecars: {', '.join(missing_sidecars)}."
        )
    return model_path


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve SOMA model directory, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("soma")

    if model_path is not None:
        model_path = Path(model_path)
        if model_path.is_dir() and (model_path / SOMA_CORE_ASSET).exists() and _missing_assets(model_path):
            return download_model(model_path)
        return validate_path(model_path)

    cache_path = get_cache_dir() / "soma"
    if not _missing_assets(cache_path):
        return validate_path(cache_path)

    return download_model()


def download_model(model_dir: PathLike | None = None) -> Path:
    """Download SOMA assets from Hugging Face."""
    cache_dir = Path(model_dir) if model_dir is not None else get_cache_dir() / "soma"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SOMA_ASSETS if not (cache_dir / name).exists()]
    if missing:
        print(f"Downloading SOMA model to {cache_dir}...")
        for name in missing:
            download_file(f"{SOMA_BASE_URL}/{name}", cache_dir / name)
        print("Done")
    return validate_path(cache_dir)


def ensure_identity_assets(model_dir: Path, model_type: str) -> None:
    """Ensure supplementary SOMA assets exist for a given identity backend."""
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
            download_file(f"{SOMA_BASE_URL}/{name}", path)
        print("Done")


def preprocess_model(upstream_dir: PathLike, output_dir: PathLike) -> Path:
    """Generate normalized SOMA assets from upstream SOMA-X 0.2.1 assets."""
    upstream_dir = Path(upstream_dir)
    output_dir = Path(output_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("SOMA preprocessing requires `uv` on PATH.")

    script = Path(__file__).with_name("generate_asset.py")
    command = [uv, "run", "--no-project", str(script), str(upstream_dir), str(output_dir)]
    print("SOMA: preprocessing upstream 0.2.1 assets with usd-core via uv.")
    print(f"SOMA: running {' '.join(command)}")
    subprocess.run(command, check=True)
    output_file = output_dir / SOMA_CORE_ASSET
    if not output_file.is_file():
        raise RuntimeError(f"SOMA preprocessing did not produce {output_file}")
    return output_dir


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
    if spec.requires_direct_file and not path.is_file():
        raise FileNotFoundError(f"SOMA {normalized} identity model file not found: {path}")
    if spec.requires_direct_file and path.suffix not in {".pkl", ".npz"}:
        raise ValueError(f"Expected a SOMA {normalized} identity .pkl or .npz file, got: {path}")
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


def compute_sparse_skin_weights(skin_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = (np.abs(skin_weights) > 1e-8).sum(axis=1)
    indices = np.full((skin_weights.shape[0], int(counts.max())), -1, dtype=np.int64)
    weights = np.zeros(indices.shape, dtype=skin_weights.dtype)
    for vertex, row in enumerate(skin_weights):
        active = np.flatnonzero(np.abs(row) > 1e-8)
        indices[vertex, : len(active)] = active
        weights[vertex, : len(active)] = row[active]
    return indices, weights


def _dense_skin_weights(rig_data: dict[str, Any]) -> np.ndarray:
    weights = csc_matrix(
        (
            rig_data["skinning_weights_data"],
            rig_data["skinning_weights_indices"],
            rig_data["skinning_weights_indptr"],
        ),
        shape=tuple(int(x) for x in rig_data["skinning_weights_shape"]),
    ).toarray()
    return np.asarray(weights, dtype=np.float32)


def _prefixed_rig_data(data: Any, prefix: str) -> dict[str, Any] | None:
    names = [f"{prefix}{name}" for name in SOMA_LEGACY_NPZ_FIELDS]
    if not all(name in data for name in names):
        return None
    return {name: data[f"{prefix}{name}"] for name in SOMA_LEGACY_NPZ_FIELDS}


def _procedural_rig_data(data: Any) -> SomaProceduralRig | None:
    if not all(name in data for name in SOMA_PROCEDURAL_NPZ_FIELDS):
        return None
    return SomaProceduralRig(
        public_joint_indices_full=np.asarray(data["procedural_public_joint_indices_full"], dtype=np.int64),
        rotation_matrix=np.asarray(data["procedural_rotation_matrix"], dtype=np.float32),
        translation_matrix=np.asarray(data["procedural_translation_matrix"], dtype=np.float32),
        source_axis_ids=np.asarray(data["procedural_source_axis_ids"], dtype=np.int64),
        source_axis_signs=np.asarray(data["procedural_source_axis_signs"], dtype=np.float32),
        twist_joint_indices=np.asarray(data["procedural_twist_joint_indices"], dtype=np.int64),
        twist_axis_ids=np.asarray(data["procedural_twist_axis_ids"], dtype=np.int64),
        twist_axis_signs=np.asarray(data["procedural_twist_axis_signs"], dtype=np.float32),
    )


def active_public_skin_weights(data: SomaWeights, vertex_map: np.ndarray | None) -> np.ndarray | None:
    if data.public is None:
        return None
    if vertex_map is None:
        return data.public.skin_weights_full
    return data.public.skin_weights_full[vertex_map]


def public_joint_metadata(data: SomaWeights) -> tuple[list[int], list[str]]:
    if data.public is None:
        return [parent - 1 for parent in data.topology.parents_full[1:]], data.joint_names_full[1:]

    return [parent - 1 for parent in data.public.parents_full[1:]], data.public.joint_names_full[1:]


def _missing_assets(model_dir: Path) -> list[str]:
    return [name for name in SOMA_ASSETS if not (model_dir / name).exists()]


def _missing_legacy_npz_fields(model_dir: Path) -> list[str]:
    core_asset = model_dir / SOMA_CORE_ASSET
    if not core_asset.exists():
        return []
    with np.load(core_asset, allow_pickle=False) as data:
        return [name for name in SOMA_LEGACY_NPZ_FIELDS if name not in data]


def _preprocess_required_message(model_path: Path, missing_fields: list[str]) -> str:
    output_dir = model_path / "preprocessed"
    return (
        f"SOMA model path {model_path} has upstream 0.2.1 assets but {SOMA_CORE_ASSET} is missing normalized "
        f"rig fields: {', '.join(missing_fields)}. Run "
        f"`body-models preprocess-soma {model_path} {output_dir}`."
    )


def _soma_preprocessed_cache_dir() -> Path:
    preprocessed_dir = get_cache_dir() / "soma" / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return preprocessed_dir


def _identity_transfer_cache_file(asset_dir: Path, model_type: str) -> Path:
    preprocessed_dir = _soma_preprocessed_cache_dir()
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


@lru_cache(maxsize=None)
def load_identity_transfer_data(asset_dir: Path, model_type: str) -> SomaIdentityTransfer:
    normalized = model_type.lower()
    spec = MODEL_TYPE_SPECS.get(normalized)
    if spec is None or spec.asset_dir is None or spec.source_mesh_name is None or spec.target_mesh_name is None:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}")

    cache_file = _identity_transfer_cache_file(asset_dir, model_type)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            return SomaIdentityTransfer(
                source_vertices=np.asarray(data["source_vertices"], dtype=np.float32).copy(),
                source_tetrahedra=np.asarray(data["source_tetrahedra"], dtype=np.int64).copy(),
                face_ids=np.asarray(data["face_ids"], dtype=np.int64).copy(),
                bary_coords=np.asarray(data["bary_coords"], dtype=np.float32).copy(),
                unknown_ids=np.asarray(data["unknown_ids"], dtype=np.int64).copy(),
                anchor_ids=np.asarray(data["anchor_ids"], dtype=np.int64).copy(),
                solve_matrix=np.asarray(data["solve_matrix"], dtype=np.float32).copy(),
                anchor_matrix=np.asarray(data["anchor_matrix"], dtype=np.float32).copy(),
                rhs_base=np.asarray(data["rhs_base"], dtype=np.float32).copy(),
                internal_to_source_rotation=np.eye(3, dtype=np.float32),
                internal_to_source_translation=np.zeros(3, dtype=np.float32),
                source_to_soma_rotation=np.eye(3, dtype=np.float32),
                source_scale=spec.source_scale,
                output_scale=spec.output_scale,
            )

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
        facial_inner_vertices = load_model_data(asset_dir).facial_inner_vertices
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

    return SomaIdentityTransfer(
        source_vertices=source_vertices,
        source_tetrahedra=source_tetrahedra,
        face_ids=face_ids,
        bary_coords=bary_coords,
        unknown_ids=unknown_ids,
        anchor_ids=anchor_ids,
        solve_matrix=solve_matrix,
        anchor_matrix=anchor_matrix,
        rhs_base=rhs_base,
        internal_to_source_rotation=np.eye(3, dtype=np.float32),
        internal_to_source_translation=np.zeros(3, dtype=np.float32),
        source_to_soma_rotation=np.eye(3, dtype=np.float32),
        source_scale=spec.source_scale,
        output_scale=spec.output_scale,
    )


def _correctives_cache_file(asset_dir: Path) -> Path:
    preprocessed_dir = _soma_preprocessed_cache_dir()
    key = hashlib.md5(f"v3:{(asset_dir / SOMA_CORRECTIVES_ASSET).resolve()}".encode()).hexdigest()
    return preprocessed_dir / f"correctives_{key}.npz"


def _joint_regressor_cache_file(asset_dir: Path, joint_count: int) -> Path:
    preprocessed_dir = _soma_preprocessed_cache_dir()
    asset_path = asset_dir / SOMA_CORE_ASSET
    stat = asset_path.stat()
    key = hashlib.md5(
        f"joint-regressor-v2:{joint_count}:{asset_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    return preprocessed_dir / f"joint_regressor_{key}.npz"


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


def _load_pose_correctives_weights(asset_dir: Path) -> SomaCorrectives:
    """Load SOMA pose-corrective weights in backend-agnostic form."""
    cache_file = _correctives_cache_file(asset_dir)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            if bool(data["use_tanh"][0]):
                raise ValueError(f"Unsupported SOMA corrective cache with tanh activation: {cache_file}")
            return SomaCorrectives(
                corrective_bindpose=np.asarray(data["bindpose"], dtype=np.float32).copy(),
                corrective_W1=np.asarray(data["W1"], dtype=np.float32).copy(),
                corrective_W2_rows=np.asarray(data["W2_rows"], dtype=np.int64).copy(),
                corrective_W2_cols=np.asarray(data["W2_cols"], dtype=np.int64).copy(),
                corrective_W2_values=np.asarray(data["W2_values"], dtype=np.float32).copy(),
            )

    checkpoint_path = asset_dir / SOMA_CORRECTIVES_ASSET
    ckpt = _load_sparse_checkpoint_numpy(checkpoint_path)
    if bool(ckpt["use_tanh"]):
        raise ValueError(f"Unsupported SOMA corrective checkpoint with tanh activation: {checkpoint_path}")

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
    np.savez_compressed(
        cache_file,
        bindpose=bindpose,
        W1=W1,
        W2_rows=W2_rows,
        W2_cols=W2_cols,
        W2_values=W2_values,
        num_vertices=np.array([num_vertices], dtype=np.int64),
        use_tanh=np.array([False], dtype=np.bool_),
    )

    return SomaCorrectives(
        corrective_bindpose=bindpose.copy(),
        corrective_W1=W1.copy(),
        corrective_W2_rows=W2_rows.copy(),
        corrective_W2_cols=W2_cols.copy(),
        corrective_W2_values=W2_values.copy(),
    )


def _get_joint_children_ids(parents: np.ndarray) -> list[list[int]]:
    parent_ids = parents.tolist()
    children = [[] for _ in range(len(parent_ids))]
    for i in range(1, len(parent_ids)):
        children[parent_ids[i]].append(i)
    return children


def _pairwise_dist(a: Float[np.ndarray, "A D"], b: Float[np.ndarray, "B D"]) -> Float[np.ndarray, "A B"]:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _get_basis_weights(
    control_points: Float[np.ndarray, "C 3"],
    query_point: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "C"]:
    """Compute dense linear-RBF interpolation weights for one query point."""
    num_points, dim = control_points.shape

    K = _pairwise_dist(control_points, control_points).astype(np.float64, copy=False)
    K[np.diag_indices(num_points)] += 1e-8

    ones = np.ones((num_points, 1), dtype=np.float64)
    P = np.concatenate([ones, control_points.astype(np.float64, copy=False)], axis=1)
    Z = np.zeros((dim + 1, dim + 1), dtype=np.float64)
    A = np.block([[K, P], [P.T, Z]])

    r = np.linalg.norm(control_points - query_point[None, :], axis=1)
    rhs = np.concatenate(
        [
            r.astype(np.float64, copy=False),
            np.array([1.0], dtype=np.float64),
            query_point.astype(np.float64, copy=False),
        ]
    )
    lu, piv = scipy_linalg.lu_factor(A)
    weights = scipy_linalg.lu_solve((lu, piv), rhs)
    return weights[:num_points].astype(np.float32, copy=False)


def _build_joint_position_regressor(
    bind_shape: Float[np.ndarray, "V 3"],
    bind_world_transforms: Float[np.ndarray, "J 4 4"],
    skin_weights: Float[np.ndarray, "V J"],
    joint_parents: Int[np.ndarray, "J"],
    vertex_ids_to_exclude: Int[np.ndarray, "N"] | None,
) -> Float[np.ndarray, "J V"]:
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


def _load_or_build_joint_position_regressor(
    asset_dir: Path,
    bind_shape: Float[np.ndarray, "V 3"],
    bind_world_transforms: Float[np.ndarray, "J 4 4"],
    skin_weights: Float[np.ndarray, "V J"],
    joint_parents: Int[np.ndarray, "J"],
    vertex_ids_to_exclude: Int[np.ndarray, "N"] | None,
) -> Float[np.ndarray, "J V"]:
    cache_file = _joint_regressor_cache_file(asset_dir, bind_world_transforms.shape[0])
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            return np.asarray(data["joint_regressor"], dtype=np.float32).copy()

    joint_regressor = _build_joint_position_regressor(
        bind_shape=bind_shape,
        bind_world_transforms=bind_world_transforms,
        skin_weights=skin_weights,
        joint_parents=joint_parents,
        vertex_ids_to_exclude=vertex_ids_to_exclude,
    )
    np.savez(cache_file, joint_regressor=joint_regressor)
    return joint_regressor


@lru_cache(maxsize=None)
def _load_model_data_cached(model_dir: str) -> SomaWeights:
    asset_dir = Path(model_dir)
    correctives = _load_pose_correctives_weights(asset_dir)
    with np.load(asset_dir / SOMA_CORE_ASSET, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        num_vertices = mean.shape[0]
        shapedirs = np.asarray(data["shapedirs"], dtype=np.float32).reshape(-1, num_vertices, 3)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float32)
        faces = np.asarray(data["triangles"], dtype=np.int64)
        if missing_soma_fields := [name for name in SOMA_LEGACY_NPZ_FIELDS if name not in data]:
            missing_sidecars = [name for name in SOMA_UPSTREAM_02_ASSETS if not (asset_dir / name).exists()]
            if missing_sidecars:
                sidecar_message = f" Missing upstream 0.2.1 sidecars: {', '.join(missing_sidecars)}."
            else:
                raise FileNotFoundError(_preprocess_required_message(asset_dir, missing_soma_fields))
            raise FileNotFoundError(
                f"SOMA asset {asset_dir / SOMA_CORE_ASSET} is missing rig fields: "
                f"{', '.join(missing_soma_fields)}.{sidecar_message}"
            )
        rig_data = {name: data[name] for name in SOMA_LEGACY_NPZ_FIELDS}
        public_rig_data = _prefixed_rig_data(data, "public_")
        procedural = _procedural_rig_data(data)

        bind_shape = np.asarray(rig_data["bind_shape"], dtype=np.float32)
        bind_pose_world = np.asarray(rig_data["bind_pose_world"], dtype=np.float32)
        bind_pose_local = np.asarray(rig_data["bind_pose_local"], dtype=np.float32)
        t_pose_world = np.asarray(rig_data["t_pose_world"], dtype=np.float32)
        t_pose_local = np.asarray(rig_data["t_pose_local"], dtype=np.float32)
        joint_parents_full = np.asarray(rig_data["joint_parent_ids"], dtype=np.int64)
        joint_names_full = [str(name) for name in rig_data["joint_names"]]

        skin_weights = _dense_skin_weights(rig_data)
        if (public_rig_data is None) != (procedural is None):
            raise ValueError(f"SOMA asset {asset_dir / SOMA_CORE_ASSET} has incomplete public rig metadata.")

        facial_inner = np.concatenate(
            [
                np.asarray(data["segment_eye_bags"], dtype=np.int64),
                np.asarray(data["segment_mouth_bag"], dtype=np.int64),
            ]
        )
    joint_regressor = _load_or_build_joint_position_regressor(
        asset_dir=asset_dir,
        bind_shape=bind_shape,
        bind_world_transforms=bind_pose_world,
        skin_weights=skin_weights,
        joint_parents=joint_parents_full,
        vertex_ids_to_exclude=facial_inner,
    )
    public = None
    if public_rig_data is not None and procedural is not None:
        public_skin_weights = _dense_skin_weights(public_rig_data)
        public_parents = np.asarray(public_rig_data["joint_parent_ids"], dtype=np.int64)
        public_parents_full = public_parents.astype(np.int64).tolist()
        public_joint_children_full = _get_joint_children_ids(public_parents)
        public_skinned_vertex_indices_full = [
            np.where(public_skin_weights[:, joint_index] > 0.01)[0].astype(np.int64).tolist()
            for joint_index in range(public_skin_weights.shape[1])
        ]
        public_joint_regressor = _build_joint_position_regressor(
            bind_shape=np.asarray(public_rig_data["bind_shape"], dtype=np.float32),
            bind_world_transforms=np.asarray(public_rig_data["bind_pose_world"], dtype=np.float32),
            skin_weights=public_skin_weights,
            joint_parents=np.asarray(public_rig_data["joint_parent_ids"], dtype=np.int64),
            vertex_ids_to_exclude=facial_inner,
        )
        public = SomaPublicRig(
            joint_names_full=[str(name) for name in public_rig_data["joint_names"]],
            parents_full=public_parents_full,
            bind_pose_world=np.asarray(public_rig_data["bind_pose_world"], dtype=np.float32),
            bind_pose_local=np.asarray(public_rig_data["bind_pose_local"], dtype=np.float32),
            t_pose_world=np.asarray(public_rig_data["t_pose_world"], dtype=np.float32),
            joint_regressor=public_joint_regressor,
            skin_weights_full=public_skin_weights,
            skin_weights_active=public_skin_weights,
            topology=SomaTopology(
                parents_full=public_parents_full,
                joint_children_full=public_joint_children_full,
                joint_children_indices_full=_pad_indices(public_joint_children_full),
                skinned_vertex_indices_full=public_skinned_vertex_indices_full,
                skinned_vertex_indices_full_index=_pad_indices(public_skinned_vertex_indices_full),
                kinematic_fronts_full=compute_kinematic_fronts(public_parents),
            ),
            procedural=procedural,
        )

    joint_children_full = _get_joint_children_ids(joint_parents_full)
    skinned_vertex_indices_full = [
        np.where(skin_weights[:, joint_index] > 0.01)[0].astype(np.int64).tolist()
        for joint_index in range(skin_weights.shape[1])
    ]

    parents_full = joint_parents_full.astype(np.int64).tolist()
    skin_joint_indices, skin_joint_weights = compute_sparse_skin_weights(skin_weights)
    return SomaWeights(
        mean_full=mean,
        mean_active=mean,
        shapedirs_full=shapedirs,
        shapedirs_active=shapedirs,
        eigenvalues=eigenvalues,
        bind_shape_full=bind_shape,
        bind_pose_world=bind_pose_world,
        bind_pose_local=bind_pose_local,
        t_pose_world=t_pose_world,
        t_pose_local=t_pose_local,
        joint_regressor=joint_regressor,
        skin_weights_full=skin_weights,
        skin_weights_active=skin_weights,
        skin_joint_indices_active=skin_joint_indices,
        skin_joint_weights_active=skin_joint_weights,
        faces=faces,
        vertex_map=None,
        facial_inner_vertices=facial_inner,
        topology=SomaTopology(
            parents_full=parents_full,
            joint_children_full=joint_children_full,
            joint_children_indices_full=_pad_indices(joint_children_full),
            skinned_vertex_indices_full=skinned_vertex_indices_full,
            skinned_vertex_indices_full_index=_pad_indices(skinned_vertex_indices_full),
            kinematic_fronts_full=compute_kinematic_fronts(joint_parents_full),
        ),
        correctives=correctives,
        joint_names_full=joint_names_full,
        public=public,
    )


def _pad_indices(indices: list[list[int]]) -> Int[np.ndarray, "J K"]:
    out = np.zeros((len(indices), max(map(len, indices))), dtype=np.int64)
    for index, values in enumerate(indices):
        out[index, : len(values)] = values
    return out


def load_model_data(model_path: Path) -> SomaWeights:
    """Load SOMA model data from disk."""
    model_path = Path(model_path).resolve()
    return _load_model_data_cached(str(model_path))
