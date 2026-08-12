"""Derive and cache SOMA data that is computed from the raw assets."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from jaxtyping import Float, Int, Shaped
from ptloader import load as load_pytorch_checkpoint
from scipy import linalg as scipy_linalg
from scipy import sparse as scipy_sparse

from body_models._cache import derived_cache_key, get_cache_dir, write_npz_atomic
from body_models._common import sparse
from body_models.soma._schema import (
    MODEL_TYPE_SPECS,
    SOMA_CORE_ASSET,
    SOMA_CORRECTIVES_ASSET,
    SomaCorrectives,
    SomaIdentityTransfer,
    _SparseCoo,
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
    # Imported here because the loader in `_io` builds its data through this module.
    from body_models.soma._io import ensure_identity_assets, load_model_data

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
