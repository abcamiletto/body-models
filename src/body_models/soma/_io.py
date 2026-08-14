"""I/O utilities for SOMA model loading."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from scipy.sparse import csc_matrix

from body_models import _config as config
from body_models._cache import download_hf_archive, get_cache_dir
from body_models._common import compute_sparse_skin_weights, kinematics, simplify_mesh, sparse
from body_models._common.skinning import CompactSkinning
from body_models.soma._derive import (
    _get_joint_children_ids,
    _load_or_build_joint_position_regressor,
    _load_pose_correctives_weights,
    load_identity_transfer_data,
)
from body_models.soma._schema import (
    MODEL_TYPE_SPECS,
    SOMA_ASSETS,
    SOMA_CORE_ASSET,
    SOMA_LODS,
    SOMA_NORMALIZED_NPZ_FIELDS,
    SOMA_PROCEDURAL_RIG_FIELDS,
    SOMA_RIG_FIELDS,
    SOMA_UPSTREAM_02_ASSETS,
    SOMA_XLO_FIELDS,
    SomaAssets,
    SomaControlRig,
    SomaIdentityTransfer,
    SomaKinematics,
    SomaLodMesh,
    SomaProceduralRig,
)

PathLike = Path | str

__all__ = [
    "SomaAssets",
    "SomaControlRig",
    "SomaIdentityTransfer",
    "SomaProceduralRig",
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
    data: SomaAssets,
    *,
    mean_active: Float[np.ndarray, "Va 3"],
    shapedirs_active: Float[np.ndarray, "S Va 3"],
    skin_weights_active: Float[np.ndarray, "Va Jf"],
    faces: Int[np.ndarray, "F 3"],
    full_vertex_indices: Int[np.ndarray, "Va"] | None,
    active_vertex_indices: Int[np.ndarray, "Va"] | None,
) -> SomaAssets:
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


def with_lod_mesh(data: SomaAssets, lod: str) -> SomaAssets:
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
    data: SomaAssets,
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


@cache
def _load_model_data_cached(model_dir: str) -> SomaAssets:
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
    return SomaAssets(
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


def load_model_data(model_path: Path) -> SomaAssets:
    """Load SOMA model data from disk."""
    model_path = Path(model_path).resolve()
    return _load_model_data_cached(str(model_path))


def load_model_data_for_lod(model_path: PathLike | None, lod: str, *, simplify: float = 1.0) -> tuple[Path, SomaAssets]:
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
