"""Torch-only SOMA identity backends and topology transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..mhr.torch import MHR
from ..smpl.torch import SMPL
from ..smplx.torch import SMPLX
from ..utils import get_cache_dir
from .io import ensure_identity_assets


def _resolve_identity_model_path(*model_keys: str, filename: str) -> Path | None:
    for model_key in model_keys:
        model_path = config.get_model_path(model_key)
        if model_path is None:
            continue
        path = Path(model_path)
        if path.is_dir():
            npz_path = path / f"{filename}.npz"
            if npz_path.exists():
                return npz_path
            pkl_path = path / f"{filename}.pkl"
            if pkl_path.exists():
                return pkl_path
        else:
            return path
    return None


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def _correspondence_cache_file(asset_dir: Path, model_type: str) -> Path:
    cache_dir = get_cache_dir() / "soma" / "preprocessed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"identity-v2:{model_type}:{asset_dir.resolve()}".encode()).hexdigest()
    return cache_dir / f"identity_transfer_{key}.npz"


def _load_or_compute_correspondence(
    *,
    asset_dir: Path,
    model_type: str,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_file = _correspondence_cache_file(asset_dir, model_type)
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            return (
                np.asarray(data["face_ids"], dtype=np.int64),
                np.asarray(data["bary_coords"], dtype=np.float32),
                np.asarray(data["source_tetrahedra"], dtype=np.int64),
            )

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            f"SOMA model_type={model_type!r} requires trimesh to precompute topology transfer."
        ) from exc

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

    np.savez_compressed(
        cache_file,
        face_ids=face_ids,
        bary_coords=bary_coords,
        source_tetrahedra=source_tetrahedra,
    )
    return face_ids, bary_coords, source_tetrahedra


def _cotangent_weights(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    def _cotangent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dot = (a * b).sum(dim=-1)
        cross = torch.cross(a, b, dim=-1)
        return dot / (torch.linalg.norm(cross, dim=-1) + 1e-8)

    cot0 = _cotangent(e1, e2)
    cot1 = _cotangent(e2, e0)
    cot2 = _cotangent(e0, e1)

    i_idx = torch.cat([faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1]])
    j_idx = torch.cat([faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0]])
    values = torch.cat([cot0, cot0, cot1, cot1, cot2, cot2])
    return i_idx, j_idx, values


def _build_cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    n_vertices = vertices.shape[0]
    i_idx, j_idx, values = _cotangent_weights(vertices, faces)
    weights = torch.sparse_coo_tensor(
        torch.stack([i_idx, j_idx]),
        values,
        (n_vertices, n_vertices),
        device=vertices.device,
        dtype=vertices.dtype,
    ).coalesce()
    weights = ((weights + weights.t()) / 2).coalesce()

    row_sums = torch.sparse.sum(weights, dim=1).to_dense()
    diag = torch.arange(n_vertices, device=vertices.device)
    laplacian = torch.sparse_coo_tensor(
        torch.cat([torch.stack([diag, diag]), weights.indices()], dim=1),
        torch.cat([row_sums, -weights.values()]),
        (n_vertices, n_vertices),
        device=vertices.device,
        dtype=vertices.dtype,
    ).coalesce()
    return laplacian.to_sparse_csr()


def _empty_sparse_csr(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(0, dtype=dtype),
        size=(0, 0),
        dtype=dtype,
    )


class _RestShapeTransfer(nn.Module):
    _source_faces: torch.Tensor
    _source_tetrahedra: torch.Tensor
    _target_faces: torch.Tensor
    _face_ids: torch.Tensor
    _bary_coords: torch.Tensor
    _target_vertices: torch.Tensor
    _free_vertex_ids: torch.Tensor
    _anchor_vertex_ids: torch.Tensor
    _laplacian_rhs: torch.Tensor
    _laplacian_cholesky: torch.Tensor
    _unknown_vertex_ids: torch.Tensor

    def __init__(
        self,
        *,
        source_vertices: np.ndarray,
        source_faces: np.ndarray,
        target_vertices: np.ndarray,
        target_faces: np.ndarray,
        free_vertex_ids: np.ndarray,
        asset_dir: Path,
        model_type: str,
    ) -> None:
        super().__init__()
        face_ids, bary_coords, source_tetrahedra = _load_or_compute_correspondence(
            asset_dir=asset_dir,
            model_type=model_type,
            source_vertices=source_vertices,
            source_faces=source_faces,
            target_vertices=target_vertices,
        )
        self.register_buffer("_source_faces", torch.as_tensor(source_faces, dtype=torch.int64))
        self.register_buffer("_source_tetrahedra", torch.as_tensor(source_tetrahedra, dtype=torch.int64))
        self.register_buffer("_target_faces", torch.as_tensor(target_faces, dtype=torch.int64))
        self.register_buffer("_face_ids", torch.as_tensor(face_ids, dtype=torch.int64))
        self.register_buffer("_bary_coords", torch.as_tensor(bary_coords, dtype=torch.float32))
        self.register_buffer("_target_vertices", torch.as_tensor(target_vertices, dtype=torch.float32))
        self.register_buffer("_free_vertex_ids", torch.as_tensor(free_vertex_ids, dtype=torch.int64))
        self.register_buffer("_anchor_vertex_ids", torch.empty(0, dtype=torch.int64), persistent=False)
        self._laplacian = _empty_sparse_csr()
        self._laplacian_fg = _empty_sparse_csr()
        self.register_buffer("_laplacian_rhs", torch.empty((0, 3), dtype=torch.float32), persistent=False)
        self.register_buffer("_laplacian_cholesky", torch.empty((0, 0), dtype=torch.float32), persistent=False)
        self.register_buffer("_unknown_vertex_ids", torch.empty(0, dtype=torch.int64), persistent=False)
        self._laplacian_sign = 1
        self._setup_laplacian()

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self._laplacian = fn(self._laplacian)
        self._laplacian_fg = fn(self._laplacian_fg)
        return self

    def _setup_laplacian(self) -> None:
        target_vertices = self._target_vertices
        target_faces = self._target_faces
        laplacian = _build_cotangent_laplacian(target_vertices, target_faces)
        self._laplacian = laplacian

        n_vertices = int(target_vertices.shape[0])
        unknown_ids = self._free_vertex_ids
        anchor_mask = torch.ones(n_vertices, dtype=torch.bool, device=target_vertices.device)
        anchor_mask[unknown_ids] = False
        anchor_ids = torch.where(anchor_mask)[0]

        lap_coo = laplacian.to_sparse_coo()
        indices = lap_coo.indices()
        values = lap_coo.values()

        n_unknown = int(unknown_ids.numel())
        n_anchor = int(anchor_ids.numel())
        row_mapping = torch.zeros(n_vertices, dtype=torch.int64, device=target_vertices.device)
        row_mapping[unknown_ids] = torch.arange(n_unknown, dtype=torch.int64, device=target_vertices.device)
        col_unknown_mapping = row_mapping
        col_anchor_mapping = torch.zeros(n_vertices, dtype=torch.int64, device=target_vertices.device)
        col_anchor_mapping[anchor_ids] = torch.arange(n_anchor, dtype=torch.int64, device=target_vertices.device)

        row_mask = torch.isin(indices[0], unknown_ids)
        unknown_indices = indices[:, row_mask].clone()
        unknown_values = values[row_mask]
        unknown_indices[0] = row_mapping[unknown_indices[0]]

        lap_u = torch.sparse_coo_tensor(
            unknown_indices,
            unknown_values,
            (n_unknown, n_vertices),
            device=target_vertices.device,
            dtype=target_vertices.dtype,
        ).coalesce().to_sparse_csr()
        self._laplacian_rhs = torch.sparse.mm(lap_u, target_vertices)

        ff_mask = torch.isin(unknown_indices[1], unknown_ids)
        fg_mask = torch.isin(unknown_indices[1], anchor_ids)

        ff_indices = unknown_indices[:, ff_mask].clone()
        ff_indices[1] = col_unknown_mapping[ff_indices[1]]
        ff = torch.sparse_coo_tensor(
            ff_indices,
            unknown_values[ff_mask],
            (n_unknown, n_unknown),
            device=target_vertices.device,
            dtype=target_vertices.dtype,
        ).coalesce().to_sparse_csr()

        fg_indices = unknown_indices[:, fg_mask].clone()
        fg_indices[1] = col_anchor_mapping[fg_indices[1]]
        self._laplacian_fg = torch.sparse_coo_tensor(
            fg_indices,
            unknown_values[fg_mask],
            (n_unknown, n_anchor),
            device=target_vertices.device,
            dtype=target_vertices.dtype,
        ).coalesce().to_sparse_csr()

        ff_dense = ff.to_dense()
        self._laplacian_sign = -1
        self._laplacian_cholesky = torch.linalg.cholesky(-ff_dense)
        self._unknown_vertex_ids = unknown_ids
        self._anchor_vertex_ids = anchor_ids

    def _barycentric_transfer(self, source_vertices: torch.Tensor) -> torch.Tensor:
        has_batch = source_vertices.ndim == 3
        if not has_batch:
            source_vertices = source_vertices[None]

        source_faces = self._source_faces.to(device=source_vertices.device)
        source_tetrahedra = self._source_tetrahedra.to(device=source_vertices.device)
        face_ids = self._face_ids.to(device=source_vertices.device)
        bary_coords = self._bary_coords.to(device=source_vertices.device, dtype=source_vertices.dtype)

        f0 = source_vertices[:, source_faces[:, 0]]
        f1 = source_vertices[:, source_faces[:, 1]]
        f2 = source_vertices[:, source_faces[:, 2]]
        fabricated = f0 + torch.cross(f1 - f0, f2 - f0, dim=-1)
        source_vertices_tet = torch.cat([source_vertices, fabricated], dim=1)

        tet_indices = source_tetrahedra[face_ids]
        v0 = source_vertices_tet[:, tet_indices[:, 0]]
        v1 = source_vertices_tet[:, tet_indices[:, 1]]
        v2 = source_vertices_tet[:, tet_indices[:, 2]]
        v3 = source_vertices_tet[:, tet_indices[:, 3]]
        bc = bary_coords[None]
        result = v0 * bc[..., 0:1] + v1 * bc[..., 1:2] + v2 * bc[..., 2:3] + v3 * bc[..., 3:4]
        return result if has_batch else result[0]

    def _laplacian_blend(self, target_vertices: torch.Tensor) -> torch.Tensor:
        single = target_vertices.ndim == 2
        if single:
            target_vertices = target_vertices[None]

        unknown_ids = self._unknown_vertex_ids.to(device=target_vertices.device)
        anchor_ids = self._anchor_vertex_ids.to(device=target_vertices.device)
        rhs_base = self._laplacian_rhs.to(device=target_vertices.device, dtype=target_vertices.dtype)
        lap_fg = self._laplacian_fg.to(device=target_vertices.device)
        chol = self._laplacian_cholesky.to(device=target_vertices.device, dtype=target_vertices.dtype)
        n_unknown = int(unknown_ids.numel())
        n_anchor = int(anchor_ids.numel())

        x_anchor = target_vertices[:, anchor_ids]
        x_anchor_2d = x_anchor.permute(1, 0, 2).reshape(n_anchor, -1)
        rhs = rhs_base.repeat(1, target_vertices.shape[0]) - torch.sparse.mm(lap_fg, x_anchor_2d)
        x_unknown = torch.cholesky_solve(self._laplacian_sign * rhs, chol)
        x_unknown = x_unknown.reshape(n_unknown, target_vertices.shape[0], 3).permute(1, 0, 2)

        out = target_vertices.clone()
        out[:, unknown_ids] = x_unknown
        return out[0] if single else out

    def forward(self, source_vertices: torch.Tensor) -> torch.Tensor:
        transferred = self._barycentric_transfer(source_vertices)
        return self._laplacian_blend(transferred)


class TorchIdentityBackend(nn.Module):
    _mhr_model: MHR | None
    _smpl_like_model: SMPL | SMPLX | None

    def __init__(self, model_type: str, soma_asset_dir: Path, facial_inner_vertices: np.ndarray) -> None:
        super().__init__()
        normalized = model_type.lower()
        if normalized not in {"mhr", "smpl", "smplx"}:
            raise ValueError(
                f"Unsupported SOMA model_type: {model_type}. Supported identity backends are soma, mhr, smpl, smplx."
            )
        self.model_type = normalized
        self._mhr_model = None
        self._smpl_like_model = None

        ensure_identity_assets(soma_asset_dir, normalized)
        if normalized == "mhr":
            source_mesh_path = soma_asset_dir / "MHR" / "base_body_lod1.obj"
            target_mesh_path = soma_asset_dir / "MHR" / "SOMA_wrap_lod1.obj"
            self.identity_dim = 45
            self.scale_dim = 68
            self._source_model_scale_to_transfer = 100.0
            self._transfer_scale_to_soma_core = 1.0
            self._mhr_model = MHR(model_path=config.get_model_path("mhr"), simplify=1.0)
        else:
            source_mesh_path = soma_asset_dir / normalized.upper() / "base_body.obj"
            target_mesh_path = soma_asset_dir / normalized.upper() / "SOMA_wrap.obj"
            self.identity_dim = 10
            self.scale_dim = None
            self._source_model_scale_to_transfer = 1.0
            self._transfer_scale_to_soma_core = 100.0
            if normalized == "smplx":
                model_path = _resolve_identity_model_path("smplx-neutral", "smplx", filename="SMPLX_NEUTRAL")
                self._smpl_like_model = SMPLX(model_path=model_path, gender="neutral", simplify=1.0)
            else:
                model_path = _resolve_identity_model_path("smpl-neutral", "smpl", filename="SMPL_NEUTRAL")
                self._smpl_like_model = SMPL(model_path=model_path, gender="neutral", simplify=1.0)

        source_vertices, source_faces = _parse_obj(source_mesh_path)
        target_vertices, target_faces = _parse_obj(target_mesh_path)
        self._rest_shape_transfer = _RestShapeTransfer(
            source_vertices=source_vertices,
            source_faces=source_faces,
            target_vertices=target_vertices,
            target_faces=target_faces,
            free_vertex_ids=np.asarray(facial_inner_vertices, dtype=np.int64),
            asset_dir=soma_asset_dir,
            model_type=normalized,
        )

    def _broadcast(self, value: torch.Tensor | None, batch_size: int, dim: int, *, device, dtype) -> torch.Tensor:
        if value is None:
            return torch.zeros((batch_size, dim), device=device, dtype=dtype)
        if value.shape[0] == 1 and batch_size > 1:
            return value.expand(batch_size, -1)
        return value

    def _rest_shape_mhr(self, identity: torch.Tensor, scale_params: torch.Tensor | None) -> torch.Tensor:
        model = self._mhr_model
        assert model is not None
        batch_size = identity.shape[0]
        scale_params = self._broadcast(
            scale_params,
            batch_size,
            self.scale_dim or 0,
            device=identity.device,
            dtype=identity.dtype,
        )
        zero_pose = torch.zeros(
            (batch_size, model.pose_dim),
            device=identity.device,
            dtype=identity.dtype,
        )
        zero_pose[:, -scale_params.shape[1] :] = scale_params
        expression = torch.zeros((batch_size, model.EXPR_DIM), device=identity.device, dtype=identity.dtype)
        with torch.no_grad():
            rest_shape = model.forward_vertices(
                shape=identity,
                pose=zero_pose,
                expression=expression,
            )
        return rest_shape

    def _rest_shape_smplx_like(self, identity: torch.Tensor) -> torch.Tensor:
        model = self._smpl_like_model
        assert model is not None
        identity_dim = identity.shape[1]
        dirs = model.shapedirs_full[..., :identity_dim]
        return model.v_template_full[None] + torch.einsum("bi,vci->bvc", identity, dirs)

    def forward(self, identity: torch.Tensor, scale_params: torch.Tensor | None = None) -> torch.Tensor:
        if self.model_type == "mhr":
            rest_shape_source = self._rest_shape_mhr(identity, scale_params)
        else:
            rest_shape_source = self._rest_shape_smplx_like(identity)

        if self._source_model_scale_to_transfer != 1.0:
            rest_shape_source = rest_shape_source * self._source_model_scale_to_transfer

        rest_shape_soma = self._rest_shape_transfer(rest_shape_source)
        if self._transfer_scale_to_soma_core != 1.0:
            rest_shape_soma = rest_shape_soma * self._transfer_scale_to_soma_core
        return rest_shape_soma
