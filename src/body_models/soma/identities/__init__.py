"""SOMA identity setup."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from jaxtyping import Float

from ... import common
from ...common import get_namespace
from . import mhr, smpl, smplx
from ..backend import core
from ..io import SomaIdentityTransfer

IDENTITY_BACKENDS = {
    "mhr": mhr,
    "smpl": smpl,
    "smplx": smplx,
}

__all__ = [
    "IDENTITY_BACKENDS",
    "IdentityBackend",
    "load",
    "prepare",
    "prepare_backend",
]


@dataclass(frozen=True)
class IdentityBackend:
    model_type: str
    identity_dim: int
    num_scale_params: int | None
    default_identity_value: float
    prepare_identity: Any
    prepare_for_backend: Any


def load(model_type: str, spec: Any, transfer: SomaIdentityTransfer | None = None) -> IdentityBackend:
    if model_type == "soma":
        def prepare_for_backend(_backend: str) -> IdentityBackend:
            return load(model_type, spec)

        return IdentityBackend(
            model_type=model_type,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
            prepare_identity=_prepare_soma_identity,
            prepare_for_backend=prepare_for_backend,
        )

    module = IDENTITY_BACKENDS[model_type]
    model, transfer = module.prepare(transfer)
    return _transferred_identity_backend(model_type, spec, module, model, transfer)


def prepare_backend(identity_backend: IdentityBackend, backend: str) -> IdentityBackend:
    return identity_backend.prepare_for_backend(backend)


def prepare(
    *,
    backend: Any,
    data: Any,
    identity: Float[Any, "B|1 I"] | None,
    scale_params: Float[Any, "B|1 K"] | None,
    batch_size: int,
    vertex_map: Any,
    ref: Any,
    match_warp: bool,
    xp: Any,
) -> core.PreparedIdentity:
    identity, scale_params = _resolve_inputs(
        backend=backend,
        identity=identity,
        scale_params=scale_params,
        batch_size=batch_size,
        ref=ref,
        xp=xp,
    )
    return backend.prepare_identity(
        data=data,
        identity=identity,
        scale_params=scale_params,
        vertex_map=vertex_map,
        match_warp=match_warp,
        xp=xp,
    )


def _prepare_soma_identity(
    *,
    data: Any,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    vertex_map: Any,
    match_warp: bool,
    xp: Any,
) -> core.PreparedIdentity:
    return core.prepare_identity(data=data, identity=identity, match_warp=match_warp, xp=xp)


def _transferred_identity_backend(
    model_type: str,
    spec: Any,
    module: Any,
    model: Any,
    transfer: SomaIdentityTransfer,
) -> IdentityBackend:
    def prepare_identity(
        *,
        data: Any,
        identity: Float[Any, "B I"],
        scale_params: Float[Any, "B K"] | None,
        vertex_map: Any,
        match_warp: bool,
        xp: Any,
    ) -> core.PreparedIdentity:
        rest_shape_full, rest_shape_active = _rest_shape(
            module=module,
            model=model,
            transfer=transfer,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=spec.num_scale_params,
            vertex_map=vertex_map,
            xp=xp,
        )
        return core.prepare_identity_from_rest_shape(
            data=data,
            rest_shape_full=rest_shape_full,
            rest_shape_active=rest_shape_active,
            match_warp=match_warp,
            xp=xp,
        )

    def prepare_for_backend(backend: str) -> IdentityBackend:
        backend_model = module.prepare_backend_model(model, backend)
        backend_transfer = _prepare_transfer(transfer, backend)
        return _transferred_identity_backend(model_type, spec, module, backend_model, backend_transfer)

    return IdentityBackend(
        model_type=model_type,
        identity_dim=spec.identity_dim,
        num_scale_params=spec.num_scale_params,
        default_identity_value=spec.default_identity_value,
        prepare_identity=prepare_identity,
        prepare_for_backend=prepare_for_backend,
    )


def _prepare_transfer(transfer: SomaIdentityTransfer, backend: str) -> SomaIdentityTransfer:
    if backend == "numpy":
        return transfer
    if backend == "torch":
        import torch

        array = torch.as_tensor

        def index(value):
            return torch.as_tensor(value, dtype=torch.int64)
    elif backend == "jax":
        import jax.numpy as jnp

        array = jnp.asarray
        index = jnp.asarray
    else:
        raise ValueError(f"Unsupported SOMA backend: {backend}")

    return replace(
        transfer,
        source_vertices=array(transfer.source_vertices),
        source_tetrahedra=index(transfer.source_tetrahedra),
        face_ids=index(transfer.face_ids),
        bary_coords=array(transfer.bary_coords),
        unknown_ids=index(transfer.unknown_ids),
        anchor_ids=index(transfer.anchor_ids),
        solve_matrix=array(transfer.solve_matrix),
        anchor_matrix=array(transfer.anchor_matrix),
        rhs_base=array(transfer.rhs_base),
        internal_to_source_rotation=array(transfer.internal_to_source_rotation),
        internal_to_source_translation=array(transfer.internal_to_source_translation),
        source_to_soma_rotation=array(transfer.source_to_soma_rotation),
    )


def _resolve_inputs(
    *,
    backend: Any,
    identity: Float[Any, "B|1 I"] | None,
    scale_params: Float[Any, "B|1 K"] | None,
    batch_size: int,
    ref: Any,
    xp: Any,
) -> tuple[Float[Any, "B I"], Float[Any, "B K"] | None]:
    if identity is None:
        identity = common.zeros_as(ref, shape=(1, backend.identity_dim), xp=xp)
        identity = identity + backend.default_identity_value
    if identity.shape[0] == 1 and batch_size > 1:
        identity = xp.broadcast_to(identity, (batch_size, identity.shape[-1]))

    if backend.num_scale_params is None:
        if scale_params is not None:
            raise ValueError("scale_params is only supported for SOMA model_type='mhr'.")
        return identity, None

    if scale_params is None:
        scale_params = common.zeros_as(ref, shape=(1, backend.num_scale_params), xp=xp)
    if scale_params.shape[0] == 1 and batch_size > 1:
        scale_params = xp.broadcast_to(scale_params, (batch_size, scale_params.shape[-1]))
    return identity, scale_params


def _rest_shape(
    *,
    module: Any,
    model: Any,
    transfer: SomaIdentityTransfer,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int | None,
    vertex_map: Any,
    xp: Any,
) -> tuple[Float[Any, "B Vt 3"], Float[Any, "B Va 3"]]:
    source_shape = module.shape(
        identity_model=model,
        identity=identity,
        scale_params=scale_params,
        num_scale_params=num_scale_params,
        xp=xp,
    )
    rest_shape_full = _transfer_source_shape(source_shape, transfer, xp=xp)
    rest_shape_active = rest_shape_full if vertex_map is None else rest_shape_full[:, vertex_map]
    return rest_shape_full, rest_shape_active


def _transfer_source_shape(
    source_shape: Float[Any, "B Vs 3"],
    transfer: SomaIdentityTransfer,
    *,
    xp: Any,
) -> Float[Any, "B Vt 3"]:
    source_shape = _apply_rigid_transform(
        source_shape,
        rotation=transfer.internal_to_source_rotation,
        translation=transfer.internal_to_source_translation,
        xp=xp,
    )
    source_shape = source_shape * transfer.source_scale
    target_shape = _transfer_rest_shape(
        source_shape=source_shape,
        source_tetrahedra=transfer.source_tetrahedra,
        face_ids=transfer.face_ids,
        bary_coords=transfer.bary_coords,
        unknown_ids=transfer.unknown_ids,
        anchor_ids=transfer.anchor_ids,
        solve_matrix=transfer.solve_matrix,
        anchor_matrix=transfer.anchor_matrix,
        rhs_base=transfer.rhs_base,
        xp=xp,
    )
    target_shape = _apply_rigid_transform(
        target_shape,
        rotation=transfer.source_to_soma_rotation,
        xp=xp,
    )
    return target_shape * transfer.output_scale


def _transfer_rest_shape(
    source_shape: Float[Any, "B Vs 3"],
    source_tetrahedra: Any,
    face_ids: Any,
    bary_coords: Any,
    unknown_ids: Any,
    anchor_ids: Any,
    solve_matrix: Any,
    anchor_matrix: Any,
    rhs_base: Any,
    *,
    xp: Any = None,
) -> Float[Any, "B Vt 3"]:
    if xp is None:
        xp = get_namespace(source_shape)

    tetra_faces = source_tetrahedra[:, :3]
    f0 = source_shape[:, tetra_faces[:, 0]]
    f1 = source_shape[:, tetra_faces[:, 1]]
    f2 = source_shape[:, tetra_faces[:, 2]]
    fabricated = f0 + xp.linalg.cross(f1 - f0, f2 - f0)
    source_shape_tet = xp.concat([source_shape, fabricated], axis=1)

    tet_indices = source_tetrahedra[face_ids]
    v0 = source_shape_tet[:, tet_indices[:, 0]]
    v1 = source_shape_tet[:, tet_indices[:, 1]]
    v2 = source_shape_tet[:, tet_indices[:, 2]]
    v3 = source_shape_tet[:, tet_indices[:, 3]]
    bc = bary_coords[None]
    target_shape = v0 * bc[..., 0:1] + v1 * bc[..., 1:2] + v2 * bc[..., 2:3] + v3 * bc[..., 3:4]

    if unknown_ids.shape[0] == 0:
        return target_shape

    B = target_shape.shape[0]
    num_unknown = unknown_ids.shape[0]
    num_anchor = anchor_ids.shape[0]
    anchor_vertices = target_shape[:, anchor_ids]
    anchor_vertices = xp.reshape(anchor_vertices.swapaxes(0, 1), (num_anchor, B * 3))
    rhs = xp.broadcast_to(rhs_base[:, None, :], (num_unknown, B, 3)).reshape(num_unknown, B * 3)
    rhs = rhs - anchor_matrix @ anchor_vertices
    unknown_vertices = xp.linalg.solve(solve_matrix, rhs)
    unknown_vertices = xp.reshape(unknown_vertices, (num_unknown, B, 3)).swapaxes(0, 1)
    return common.set(target_shape, (slice(None), unknown_ids), unknown_vertices, xp=xp)


def _apply_rigid_transform(
    points: Float[Any, "B V 3"],
    rotation: Float[Any, "3 3"],
    translation: Float[Any, "3"] | None = None,
    *,
    xp: Any = None,
) -> Float[Any, "B V 3"]:
    if xp is None:
        xp = get_namespace(points)
    points = points @ rotation.swapaxes(-2, -1)
    if translation is not None:
        points = points + translation
    return points
