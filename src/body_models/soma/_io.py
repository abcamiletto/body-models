"""I/O utilities for SOMA model loading."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from jaxtyping import Float, Int, Shaped
from ptloader import load as load_pytorch_checkpoint
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse
from scipy.sparse import csc_matrix

from body_models import _config as config
from body_models._cache import derived_cache_key, download_hf_archive, get_cache_dir, write_npz_atomic
from body_models._common import compute_sparse_skin_weights, kinematics, simplify_mesh, sparse
from body_models._common.skinning import CompactSkinning

PathLike = Path | str

SOMA_CORE_ASSET = "SOMA_neutral.npz"
SOMA_CORRECTIVES_ASSET = "correctives_model.pt"
SOMA_TEMPLATE_RIG_ASSET = "SOMA_template_rig.usda"
SOMA_PROCEDURAL_TRANSFORMS_ASSET = "SOMA_procedural_transforms.json"
SOMA_ASSETS = (SOMA_CORE_ASSET, SOMA_CORRECTIVES_ASSET)
SOMA_UPSTREAM_02_ASSETS = (SOMA_TEMPLATE_RIG_ASSET, SOMA_PROCEDURAL_TRANSFORMS_ASSET)
SOMA_LODS = ("mid", "low", "xlo")
SOMA_XLO_FIELDS = (
    "lod_mid_to_xlo",
    "triangles_xlo",
    "skinning_weights_xlo_data",
    "skinning_weights_xlo_indices",
    "skinning_weights_xlo_indptr",
    "skinning_weights_xlo_shape",
)
SOMA_RIG_FIELDS = (
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
# The normalized asset schema calls the user-facing control rig "public".
SOMA_PROCEDURAL_RIG_FIELDS = (
    "public_joint_indices_full",
    "rotation_matrix",
    "translation_matrix",
    "twist_joint_indices",
    "twist_axis_ids",
    "twist_axis_signs",
    "segment_start_joint_indices",
    "segment_end_joint_indices",
    "segment_parent_joint_indices",
    "segment_reverse_indices",
    "segment_alignment_rotations",
)
SOMA_PROCEDURAL_NPZ_FIELDS = tuple(f"procedural_{name}" for name in SOMA_PROCEDURAL_RIG_FIELDS)
SOMA_NORMALIZED_NPZ_FIELDS = (
    *SOMA_RIG_FIELDS,
    *(f"public_{name}" for name in SOMA_RIG_FIELDS),
    *SOMA_PROCEDURAL_NPZ_FIELDS,
)

__all__ = [
    "SomaControlRig",
    "SomaIdentityTransfer",
    "SomaProceduralRig",
    "SomaWeights",
    "download_model",
    "get_model_path",
    "load_identity_transfer_data",
    "load_model_data",
    "load_model_data_for_lod",
    "preprocess_model",
    "simplify_mesh",
    "with_active_mesh",
    "with_lod_mesh",
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
    hidden_weights: Float[np.ndarray, "input hidden"]
    basis: sparse.SparseMatrix


@dataclass(frozen=True)
class SomaKinematics:
    """Tree traversal and orientation-gather state for one SOMA rig."""

    kinematic_tree: kinematics.KinematicTree
    orientation_parent_indices: Int[np.ndarray, "Jf"]


@dataclass(frozen=True)
class SomaProceduralRig:
    control_joint_indices_full: Int[np.ndarray, "Jp"]
    rotation_matrix: Float[np.ndarray, "T Jp"]
    translation_matrix: Float[np.ndarray, "Jf Jf"]
    twist_joint_indices: Int[np.ndarray, "T"]
    twist_axis_ids: Int[np.ndarray, "T"]
    twist_axis_signs: Float[np.ndarray, "T"]
    segment_start_joint_indices: Int[np.ndarray, "S"]
    segment_end_joint_indices: Int[np.ndarray, "S"]
    segment_parent_joint_indices: Int[np.ndarray, "S"]
    segment_reverse_indices: Int[np.ndarray, "R"]
    segment_alignment_rotations: Float[np.ndarray, "S 3 3"]


@dataclass(frozen=True)
class SomaControlRig:
    joint_names_full: list[str]
    bind_pose_world: Float[np.ndarray, "Jp 4 4"]
    bind_pose_local: Float[np.ndarray, "Jp 4 4"]
    t_pose_world: Float[np.ndarray, "Jp 4 4"]
    t_pose_local: Float[np.ndarray, "Jp 4 4"]
    joint_regressor: Float[np.ndarray, "Jp Vf"]
    skin_weights_full: Float[np.ndarray, "Vf Jp"]
    skin_weights_active: Float[np.ndarray, "Va Jp"]
    kinematics: SomaKinematics
    joint_children_full: list[list[int]]
    joint_children_indices_full: Int[np.ndarray, "Jp C"]
    skinned_vertex_indices_full: list[list[int]]
    skinned_vertex_indices_full_index: Int[np.ndarray, "Jp K"]
    procedural: SomaProceduralRig


@dataclass(frozen=True)
class SomaLodMesh:
    vertex_map: Int[np.ndarray, "Va"]
    faces: Int[np.ndarray, "F 3"]
    skin_weights: Float[np.ndarray, "Va Jf"] | None = None


@dataclass(frozen=True)
class SomaWeights:
    """SOMA weights loaded from normalized assets."""

    mean_full: Float[np.ndarray, "Vf 3"]
    mean_active: Float[np.ndarray, "Va 3"]
    shapedirs_full: Float[np.ndarray, "S Vf 3"]
    shapedirs_active: Float[np.ndarray, "S Va 3"]
    eigenvalues: Float[np.ndarray, "S"]
    bind_shape_full: Float[np.ndarray, "Vf 3"]
    bind_pose_world: Float[np.ndarray, "Jf 4 4"]
    t_pose_world: Float[np.ndarray, "Jf 4 4"]
    skin_weights_full: Float[np.ndarray, "Vf Jf"]
    skin_weights_active: Float[np.ndarray, "Va Jf"]
    compact_skinning: CompactSkinning
    faces: Int[np.ndarray, "F 3"]
    vertex_map: Int[np.ndarray, "Va"] | None
    facial_inner_vertices: Int[np.ndarray, "Va"]
    kinematics: SomaKinematics
    correctives: SomaCorrectives
    control_rig: SomaControlRig
    lods: dict[str, SomaLodMesh] | None = None


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
    num_shape_coeffs: int
    num_scale_coeffs: int | None = None
    default_identity_value: float = 0.0
    identity_model_kwargs: dict[str, Any] = field(default_factory=dict)
    source_scale: float = 1.0
    output_scale: float = 1.0
    asset_dir: str | None = None
    source_mesh_name: str | None = None
    target_mesh_name: str | None = None
    use_laplacian: bool = True


MODEL_TYPE_SPECS = {
    "soma": _ModelTypeSpec(num_shape_coeffs=128),
    "mhr": _ModelTypeSpec(
        num_shape_coeffs=45,
        num_scale_coeffs=68,
        source_scale=100.0,
        asset_dir="MHR",
        source_mesh_name="base_body_lod1.obj",
        target_mesh_name="SOMA_wrap_lod1.obj",
    ),
    "anny": _ModelTypeSpec(
        num_shape_coeffs=6,
        default_identity_value=0.5,
        output_scale=100.0,
        asset_dir="Anny",
        identity_model_kwargs={"all_phenotypes": False},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
        use_laplacian=False,
    ),
    "smpl": _ModelTypeSpec(
        num_shape_coeffs=10,
        output_scale=100.0,
        asset_dir="SMPL",
        identity_model_kwargs={"gender": "neutral"},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
    ),
    "smplx": _ModelTypeSpec(
        num_shape_coeffs=10,
        output_scale=100.0,
        asset_dir="SMPLX",
        identity_model_kwargs={"gender": "neutral"},
        source_mesh_name="base_body.obj",
        target_mesh_name="SOMA_wrap.obj",
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
    missing_fields = _missing_normalized_npz_fields(model_path)
    if missing_fields:
        raise _missing_rig_fields_error(model_path, missing_fields)
    return model_path


def get_model_path(model_path: PathLike | None = None) -> Path:
    """Resolve SOMA model directory, downloading if necessary."""
    if model_path is None:
        model_path = config.get_model_path("soma")

    if model_path is not None:
        return validate_path(model_path)

    cache_path = get_cache_dir() / "soma"
    if not _missing_assets(cache_path):
        return validate_path(cache_path)

    return download_model()


def download_model(output_dir: PathLike | None = None) -> Path:
    """Download SOMA assets from Hugging Face."""
    output_dir = Path(output_dir) if output_dir is not None else get_cache_dir() / "soma"
    missing = [name for name in SOMA_ASSETS if not (output_dir / name).exists()]
    if missing:
        print(f"Downloading SOMA model to {output_dir}...")
        download_hf_archive("soma/assets.zip", output_dir)
        print("Done")
    return validate_path(output_dir)


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
        raise FileNotFoundError(f"SOMA {normalized} identity assets are missing from {asset_dir}: {', '.join(missing)}")


def preprocess_model(upstream_dir: PathLike, output_dir: PathLike) -> Path:
    """Generate normalized SOMA assets from upstream SOMA-X 0.2.1 assets."""
    upstream_dir = Path(upstream_dir)
    output_dir = Path(output_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("SOMA preprocessing requires `uv` on PATH.")

    script = Path(__file__).parent / "_tools" / "generate_asset.py"
    command = [uv, "run", "--no-project", str(script), str(upstream_dir), str(output_dir)]
    print("SOMA: preprocessing upstream 0.2.1 assets with usd-core via uv.")
    print(f"SOMA: running {' '.join(command)}")
    subprocess.run(command, check=True)
    output_file = output_dir / SOMA_CORE_ASSET
    if not output_file.is_file():
        raise RuntimeError(f"SOMA preprocessing did not produce {output_file}")
    return output_dir


def _dense_skin_weights(rig_data: dict[str, Any]) -> Float[np.ndarray, "V J"]:
    weights = csc_matrix(
        (
            rig_data["skinning_weights_data"],
            rig_data["skinning_weights_indices"],
            rig_data["skinning_weights_indptr"],
        ),
        shape=tuple(int(x) for x in rig_data["skinning_weights_shape"]),
    ).toarray()
    return np.asarray(weights, dtype=np.float32)


def _control_rig_data(data: Any) -> dict[str, Any]:
    return {name: data[f"public_{name}"] for name in SOMA_RIG_FIELDS}


def _procedural_rig_data(data: Any) -> SomaProceduralRig:
    dtypes = {
        "public_joint_indices_full": np.int64,
        "rotation_matrix": np.float32,
        "translation_matrix": np.float32,
        "twist_joint_indices": np.int64,
        "twist_axis_ids": np.int64,
        "twist_axis_signs": np.float32,
        "segment_start_joint_indices": np.int64,
        "segment_end_joint_indices": np.int64,
        "segment_parent_joint_indices": np.int64,
        "segment_reverse_indices": np.int64,
        "segment_alignment_rotations": np.float32,
    }
    values = {name: np.asarray(data[f"procedural_{name}"], dtype=dtypes[name]) for name in SOMA_PROCEDURAL_RIG_FIELDS}
    return SomaProceduralRig(
        control_joint_indices_full=values["public_joint_indices_full"],
        rotation_matrix=values["rotation_matrix"],
        translation_matrix=values["translation_matrix"],
        twist_joint_indices=values["twist_joint_indices"],
        twist_axis_ids=values["twist_axis_ids"],
        twist_axis_signs=values["twist_axis_signs"],
        segment_start_joint_indices=values["segment_start_joint_indices"],
        segment_end_joint_indices=values["segment_end_joint_indices"],
        segment_parent_joint_indices=values["segment_parent_joint_indices"],
        segment_reverse_indices=values["segment_reverse_indices"],
        segment_alignment_rotations=values["segment_alignment_rotations"],
    )


def with_active_mesh(
    data: SomaWeights,
    *,
    mean_active: Float[np.ndarray, "Va 3"],
    shapedirs_active: Float[np.ndarray, "S Va 3"],
    skin_weights_active: Float[np.ndarray, "Va Jf"],
    faces: Int[np.ndarray, "F 3"],
    full_vertex_indices: Int[np.ndarray, "Va"] | None,
    active_vertex_indices: Int[np.ndarray, "Va"] | None,
) -> SomaWeights:
    """Replace the active mesh using indices into the full and current meshes."""
    skin_joint_indices_active, skin_joint_weights_active = compute_sparse_skin_weights(skin_weights_active)
    skin_joint_indices_active = np.maximum(skin_joint_indices_active - 1, -1)
    control_skin_weights_active = _active_control_skin_weights(data, full_vertex_indices)
    control_rig = replace(data.control_rig, skin_weights_active=control_skin_weights_active)
    correctives = replace(
        data.correctives,
        basis=_active_corrective_basis(data.correctives.basis, active_vertex_indices),
    )
    return replace(
        data,
        mean_active=np.asarray(mean_active, dtype=np.float32),
        shapedirs_active=np.asarray(shapedirs_active, dtype=np.float32),
        skin_weights_active=np.asarray(skin_weights_active, dtype=np.float32),
        compact_skinning=CompactSkinning(
            skin_joint_indices_active,
            skin_joint_weights_active,
        ),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_map=full_vertex_indices,
        correctives=correctives,
        control_rig=control_rig,
    )


def with_lod_mesh(data: SomaWeights, lod: str) -> SomaWeights:
    normalized = lod.lower()
    if normalized not in SOMA_LODS:
        raise ValueError(f"SOMA lod must be one of {SOMA_LODS}, got {lod!r}")
    if normalized == "mid":
        return data
    if data.lods is None or normalized not in data.lods:
        raise ValueError(
            f"SOMA lod={normalized!r} requires preprocessed LOD arrays in {SOMA_CORE_ASSET}. "
            "Regenerate the hosted SOMA assets with `body-models preprocess-soma` from SOMA-X v0.2 assets."
        )
    lod_mesh = data.lods[normalized]
    skin_weights = lod_mesh.skin_weights
    if skin_weights is None:
        skin_weights = data.skin_weights_full[lod_mesh.vertex_map]
    return with_active_mesh(
        data,
        mean_active=data.mean_full[lod_mesh.vertex_map],
        shapedirs_active=data.shapedirs_full[:, lod_mesh.vertex_map],
        skin_weights_active=skin_weights,
        faces=lod_mesh.faces,
        full_vertex_indices=lod_mesh.vertex_map,
        active_vertex_indices=lod_mesh.vertex_map,
    )


def _active_control_skin_weights(
    data: SomaWeights,
    vertex_map: Int[np.ndarray, "Va"] | None,
) -> Float[np.ndarray, "Va Jp"]:
    if vertex_map is None:
        return data.control_rig.skin_weights_full
    return data.control_rig.skin_weights_full[vertex_map]


def _active_corrective_basis(
    basis: sparse.SparseMatrix,
    active_vertex_indices: Int[np.ndarray, "Va"] | None,
) -> sparse.SparseMatrix:
    if active_vertex_indices is not None:
        columns = (active_vertex_indices[:, None] * 3 + np.arange(3)).reshape(-1)
        basis = sparse.select_columns(basis, columns)
    return basis


def _missing_assets(model_dir: Path) -> list[str]:
    return [name for name in SOMA_ASSETS if not (model_dir / name).exists()]


def _missing_normalized_npz_fields(model_dir: Path) -> list[str]:
    core_asset = model_dir / SOMA_CORE_ASSET
    if not core_asset.exists():
        return []
    with np.load(core_asset, allow_pickle=False) as data:
        return [name for name in SOMA_NORMALIZED_NPZ_FIELDS if name not in data]


def _preprocess_required_message(model_path: Path, missing_fields: list[str]) -> str:
    output_dir = model_path / "preprocessed"
    return (
        f"SOMA model path {model_path} has upstream 0.2.1 assets but {SOMA_CORE_ASSET} is missing normalized "
        f"rig fields: {', '.join(missing_fields)}. Run "
        f"`body-models preprocess-soma {model_path} {output_dir}`."
    )


def _missing_rig_fields_error(asset_dir: Path, missing_fields: list[str]) -> FileNotFoundError:
    missing_sidecars = [name for name in SOMA_UPSTREAM_02_ASSETS if not (asset_dir / name).exists()]
    if not missing_sidecars:
        return FileNotFoundError(_preprocess_required_message(asset_dir, missing_fields))
    return FileNotFoundError(
        f"SOMA model path {asset_dir} is missing required NPZ fields: {', '.join(missing_fields)}. "
        f"Missing upstream 0.2.1 sidecars: {', '.join(missing_sidecars)}."
    )


def _soma_preprocessed_cache_dir() -> Path:
    preprocessed_dir = get_cache_dir() / "soma" / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    return preprocessed_dir


def _identity_transfer_cache_file(model_type: str, sources: list[Path]) -> Path:
    preprocessed_dir = _soma_preprocessed_cache_dir()
    key = derived_cache_key(
        "soma-identity-transfer-v3",
        sources=sources,
        parameters=(model_type,),
    )
    return preprocessed_dir / f"identity_transfer_{key}.npz"


def _load_mesh(path: Path) -> tuple[Float[np.ndarray, "V 3"], Int[np.ndarray, "F 3"]]:
    mesh = cast(Any, trimesh.load(path, maintain_order=True, process=False))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def _fabricate_tet(
    p0: Float[np.ndarray, "... 3"],
    p1: Float[np.ndarray, "... 3"],
    p2: Float[np.ndarray, "... 3"],
) -> Float[np.ndarray, "... 3"]:
    return p0 + np.cross(p1 - p0, p2 - p0, axis=-1)


def _compute_barycentric_coords_3d(
    p: Float[np.ndarray, "... 3"],
    v0: Float[np.ndarray, "... 3"],
    v1: Float[np.ndarray, "... 3"],
    v2: Float[np.ndarray, "... 3"],
    v3: Float[np.ndarray, "... 3"],
) -> Float[np.ndarray, "... 4"]:
    T = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
    rhs = p - v0
    b123 = np.linalg.solve(T, rhs[..., None]).squeeze(-1)
    b0 = 1.0 - b123.sum(axis=-1, keepdims=True)
    return np.concatenate([b0, b123], axis=-1).astype(np.float32, copy=False)


def _compute_identity_correspondence(
    source_vertices: Float[np.ndarray, "Vs 3"],
    source_faces: Int[np.ndarray, "Fs 3"],
    target_vertices: Float[np.ndarray, "Vt 3"],
) -> tuple[Int[np.ndarray, "Fs 4"], Int[np.ndarray, "Vt"], Float[np.ndarray, "Vt 4"]]:
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


def _build_cotangent_laplacian(
    vertices: Float[np.ndarray, "V 3"],
    faces: Int[np.ndarray, "F 3"],
) -> scipy_sparse.csr_matrix:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    def _cotangent(
        a: Float[np.ndarray, "F 3"],
        b: Float[np.ndarray, "F 3"],
    ) -> Float[np.ndarray, "F"]:
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
    target_vertices: Float[np.ndarray, "V 3"],
    target_faces: Int[np.ndarray, "F 3"],
    unknown_ids: Int[np.ndarray, "U"],
) -> tuple[
    Int[np.ndarray, "U"],
    Int[np.ndarray, "A"],
    Float[np.ndarray, "U U"],
    Float[np.ndarray, "U A"],
    Float[np.ndarray, "U 3"],
]:
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


@cache
def load_identity_transfer_data(asset_dir: Path, model_type: str) -> SomaIdentityTransfer:
    normalized = model_type.lower()
    spec = MODEL_TYPE_SPECS.get(normalized)
    if spec is None or spec.asset_dir is None or spec.source_mesh_name is None or spec.target_mesh_name is None:
        raise ValueError(f"Unsupported SOMA identity backend: {model_type}")

    ensure_identity_assets(asset_dir, normalized)
    mesh_dir = asset_dir / spec.asset_dir
    sources = [mesh_dir / spec.source_mesh_name, mesh_dir / spec.target_mesh_name]
    if spec.use_laplacian:
        sources.append(asset_dir / SOMA_CORE_ASSET)
    cache_file = _identity_transfer_cache_file(normalized, sources)
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

    write_npz_atomic(
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
    key = derived_cache_key(
        "soma-correctives-v6",
        sources=(asset_dir / SOMA_CORRECTIVES_ASSET,),
    )
    return preprocessed_dir / f"correctives_{key}.npz"


def _joint_regressor_cache_file(asset_dir: Path, joint_count: int) -> Path:
    preprocessed_dir = _soma_preprocessed_cache_dir()
    asset_path = asset_dir / SOMA_CORE_ASSET
    key = derived_cache_key(
        "soma-joint-regressor-v3",
        sources=(asset_path,),
        parameters=(joint_count,),
    )
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

    return load_pytorch_checkpoint(
        checkpoint_path,
        weights_only=True,
        pickle_global_registry={
            ("torch.serialization", "_get_layout"): _get_layout,
            ("torch._utils", "_rebuild_sparse_tensor"): _rebuild_sparse_tensor,
            ("torch", "Size"): tuple,
        },
    )


def _as_dense_float32(value: Shaped[np.ndarray, "..."] | _SparseCoo) -> Float[np.ndarray, "..."]:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    return _dense_from_sparse(value)


def _dense_from_sparse(sparse: _SparseCoo) -> Float[np.ndarray, "..."]:
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
                hidden_weights=np.asarray(data["W1"], dtype=np.float32).copy(),
                basis=sparse.scaled(_cached_sparse_matrix(data, "W2"), 0.01),
            )

    checkpoint_path = asset_dir / SOMA_CORRECTIVES_ASSET
    ckpt = _load_sparse_checkpoint_numpy(checkpoint_path)
    if bool(ckpt["use_tanh"]):
        raise ValueError(f"Unsupported SOMA corrective checkpoint with tanh activation: {checkpoint_path}")

    W1_sparse = cast(_SparseCoo, ckpt["W1"])
    W2_sparse = cast(_SparseCoo, ckpt["W2"])
    bindpose = np.asarray(cast(np.ndarray, ckpt["bindpose"]), dtype=np.float32)
    cors_per_joint = int(ckpt["C_max"])
    W1_rows = W1_sparse.indices[0].astype(np.int64, copy=False)
    W1_cols = W1_sparse.indices[1].astype(np.int64, copy=False)
    W1_values = W1_sparse.values.astype(np.float32, copy=False)
    W2_rows = W2_sparse.indices[0].astype(np.int64, copy=False)
    W2_cols = W2_sparse.indices[1].astype(np.int64, copy=False)
    W2_values = W2_sparse.values.astype(np.float32, copy=False)

    if "M1_mask" in ckpt:
        M1_mask = _as_dense_float32(cast(np.ndarray | _SparseCoo, ckpt["M1_mask"]))
        scale = np.repeat(np.repeat(M1_mask, 6, axis=0), cors_per_joint, axis=1)[W1_rows, W1_cols]
        keep = scale != 0.0
        W1_rows = W1_rows[keep]
        W1_cols = W1_cols[keep]
        W1_values = W1_values[keep] * scale[keep]

    if "M2_mask" in ckpt:
        M2_mask = _as_dense_float32(cast(np.ndarray | _SparseCoo, ckpt["M2_mask"]))
        scale = M2_mask[W2_rows // cors_per_joint, W2_cols // 3].astype(np.float32, copy=False)
        keep = scale != 0.0
        W2_rows = W2_rows[keep]
        W2_cols = W2_cols[keep]
        W2_values = W2_values[keep] * scale[keep]

    W2 = _sparse_matrix(W2_rows, W2_cols, W2_values, W2_sparse.size)
    hidden_weights = np.zeros(W1_sparse.size, dtype=np.float32)
    hidden_weights[W1_rows, W1_cols] = W1_values
    write_npz_atomic(
        cache_file,
        bindpose=bindpose,
        W1=hidden_weights,
        W2_rows=W2.row_indices,
        W2_cols=W2.column_indices,
        W2_values=W2.values,
        W2_shape=np.asarray(W2.shape, dtype=np.int64),
        use_tanh=np.array([False], dtype=np.bool_),
    )

    return SomaCorrectives(
        corrective_bindpose=bindpose.copy(),
        hidden_weights=hidden_weights,
        basis=sparse.scaled(W2, 0.01),
    )


def _cached_sparse_matrix(data: Any, name: str) -> sparse.SparseMatrix:
    return _sparse_matrix(
        np.asarray(data[f"{name}_rows"], dtype=np.int64),
        np.asarray(data[f"{name}_cols"], dtype=np.int64),
        np.asarray(data[f"{name}_values"], dtype=np.float32),
        tuple(np.asarray(data[f"{name}_shape"], dtype=np.int64).tolist()),
    )


def _sparse_matrix(
    rows: Int[np.ndarray, "NNZ"],
    columns: Int[np.ndarray, "NNZ"],
    values: Float[np.ndarray, "NNZ"],
    shape: tuple[int, ...],
) -> sparse.SparseMatrix:
    return sparse.SparseMatrix(
        row_indices=np.array(rows, dtype=np.int64, copy=True),
        column_indices=np.array(columns, dtype=np.int64, copy=True),
        values=np.array(values, dtype=np.float32, copy=True),
        shape=cast(tuple[int, int], tuple(shape)),
    )


def _get_joint_children_ids(parents: Int[np.ndarray, "J"]) -> list[list[int]]:
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
    write_npz_atomic(cache_file, compressed=False, joint_regressor=joint_regressor)
    return joint_regressor


@cache
def _load_model_data_cached(model_dir: str) -> SomaWeights:
    asset_dir = Path(model_dir)
    correctives = _load_pose_correctives_weights(asset_dir)
    with np.load(asset_dir / SOMA_CORE_ASSET, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        num_vertices = mean.shape[0]
        shapedirs = np.asarray(data["shapedirs"], dtype=np.float32).reshape(-1, num_vertices, 3)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float32)
        faces = np.asarray(data["triangles"], dtype=np.int64)
        lods = _load_lod_meshes(data)
        rig_data = {name: data[name] for name in SOMA_RIG_FIELDS}
        control_rig_data = _control_rig_data(data)
        procedural = _procedural_rig_data(data)

        bind_shape = np.asarray(rig_data["bind_shape"], dtype=np.float32)
        bind_pose_world = np.asarray(rig_data["bind_pose_world"], dtype=np.float32)
        t_pose_world = np.asarray(rig_data["t_pose_world"], dtype=np.float32)
        joint_parents_full = np.asarray(rig_data["joint_parent_ids"], dtype=np.int64)

        skin_weights = _dense_skin_weights(rig_data)
        facial_inner = np.concatenate(
            [
                np.asarray(data["segment_eye_bags"], dtype=np.int64),
                np.asarray(data["segment_mouth_bag"], dtype=np.int64),
            ]
        )
    control_skin_weights = _dense_skin_weights(control_rig_data)
    control_parents = np.asarray(control_rig_data["joint_parent_ids"], dtype=np.int64)
    control_parents_full = control_parents.astype(np.int64).tolist()
    control_joint_children_full = _get_joint_children_ids(control_parents)
    control_skinned_vertex_indices_full = [
        np.where(control_skin_weights[:, joint_index] > 0.01)[0].astype(np.int64).tolist()
        for joint_index in range(control_skin_weights.shape[1])
    ]
    control_joint_regressor = _load_or_build_joint_position_regressor(
        asset_dir=asset_dir,
        bind_shape=np.asarray(control_rig_data["bind_shape"], dtype=np.float32),
        bind_world_transforms=np.asarray(control_rig_data["bind_pose_world"], dtype=np.float32),
        skin_weights=control_skin_weights,
        joint_parents=control_parents,
        vertex_ids_to_exclude=facial_inner,
    )
    control_rig = SomaControlRig(
        joint_names_full=[str(name) for name in control_rig_data["joint_names"]],
        bind_pose_world=np.asarray(control_rig_data["bind_pose_world"], dtype=np.float32),
        bind_pose_local=np.asarray(control_rig_data["bind_pose_local"], dtype=np.float32),
        t_pose_world=np.asarray(control_rig_data["t_pose_world"], dtype=np.float32),
        t_pose_local=np.asarray(control_rig_data["t_pose_local"], dtype=np.float32),
        joint_regressor=control_joint_regressor,
        skin_weights_full=control_skin_weights,
        skin_weights_active=control_skin_weights,
        kinematics=SomaKinematics(
            kinematic_tree=kinematics.KinematicTree.from_parents(control_parents_full),
            orientation_parent_indices=control_parents,
        ),
        joint_children_full=control_joint_children_full,
        joint_children_indices_full=_pad_indices(control_joint_children_full),
        skinned_vertex_indices_full=control_skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=_pad_indices(control_skinned_vertex_indices_full),
        procedural=procedural,
    )

    parents_full = joint_parents_full.astype(np.int64).tolist()
    skin_joint_indices, skin_joint_weights = compute_sparse_skin_weights(skin_weights)
    skin_joint_indices = np.maximum(skin_joint_indices - 1, -1)
    return SomaWeights(
        mean_full=mean,
        mean_active=mean,
        shapedirs_full=shapedirs,
        shapedirs_active=shapedirs,
        eigenvalues=eigenvalues,
        bind_shape_full=bind_shape,
        bind_pose_world=bind_pose_world,
        t_pose_world=t_pose_world,
        skin_weights_full=skin_weights,
        skin_weights_active=skin_weights,
        compact_skinning=CompactSkinning(
            skin_joint_indices,
            skin_joint_weights,
        ),
        faces=faces,
        vertex_map=None,
        facial_inner_vertices=facial_inner,
        kinematics=SomaKinematics(
            kinematic_tree=kinematics.KinematicTree.from_parents(parents_full),
            orientation_parent_indices=joint_parents_full,
        ),
        correctives=correctives,
        control_rig=control_rig,
        lods=lods,
    )


def _pad_indices(indices: list[list[int]]) -> Int[np.ndarray, "J K"]:
    out = np.zeros((len(indices), max(map(len, indices))), dtype=np.int64)
    for index, values in enumerate(indices):
        out[index, : len(values)] = values
    return out


def _load_lod_meshes(data: Any) -> dict[str, SomaLodMesh] | None:
    lods: dict[str, SomaLodMesh] = {}
    if "lod_mid_to_low" in data and "triangles_low" in data:
        lods["low"] = SomaLodMesh(
            vertex_map=np.asarray(data["lod_mid_to_low"], dtype=np.int64),
            faces=np.asarray(data["triangles_low"], dtype=np.int64),
        )
    if all(name in data for name in SOMA_XLO_FIELDS):
        lods["xlo"] = SomaLodMesh(
            vertex_map=np.asarray(data["lod_mid_to_xlo"], dtype=np.int64),
            faces=np.asarray(data["triangles_xlo"], dtype=np.int64),
            skin_weights=_load_xlo_skin_weights(data),
        )
    return lods or None


def _load_xlo_skin_weights(data: Any) -> Float[np.ndarray, "Va Jf"]:
    rig_data = {
        "skinning_weights_data": data["skinning_weights_xlo_data"],
        "skinning_weights_indices": data["skinning_weights_xlo_indices"],
        "skinning_weights_indptr": data["skinning_weights_xlo_indptr"],
        "skinning_weights_shape": data["skinning_weights_xlo_shape"],
    }
    return _dense_skin_weights(rig_data)


def load_model_data(model_path: Path) -> SomaWeights:
    """Load SOMA model data from disk."""
    model_path = Path(model_path).resolve()
    return _load_model_data_cached(str(model_path))


def load_model_data_for_lod(
    model_path: PathLike | None, lod: str, *, simplify: float = 1.0
) -> tuple[Path, SomaWeights]:
    """Resolve and load SOMA data for a requested LOD and simplification level."""
    if simplify < 1.0:
        raise ValueError("simplify must be >= 1.0 (1.0 = original mesh)")
    resolved_path = get_model_path(model_path)
    data = with_lod_mesh(load_model_data(resolved_path), lod)
    if simplify == 1.0:
        return resolved_path, data
    target_faces = int(len(data.faces) / simplify)
    mean_active, faces, simplify_map = simplify_mesh(data.mean_active, data.faces.astype(int), target_faces)
    simplify_map = np.asarray(simplify_map, dtype=np.int64)
    vertex_map = simplify_map if data.vertex_map is None else data.vertex_map[simplify_map]
    weights = with_active_mesh(
        data,
        mean_active=mean_active,
        shapedirs_active=data.shapedirs_active[:, simplify_map],
        skin_weights_active=data.skin_weights_active[simplify_map],
        faces=faces,
        full_vertex_indices=vertex_map,
        active_vertex_indices=simplify_map,
    )
    return resolved_path, weights
