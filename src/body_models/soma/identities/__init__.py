"""SOMA identity setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from jaxtyping import Float

from ... import common
from ...common import get_namespace
from . import mhr, smpl, smplx
from ..io import SomaIdentityTransfer

IDENTITY_BACKENDS = {
    "mhr": mhr,
    "smpl": smpl,
    "smplx": smplx,
}

__all__ = ["IDENTITY_BACKENDS", "IdentityBackend", "load", "rest_shape"]


@dataclass(frozen=True)
class IdentityBackend:
    model_type: str
    identity_dim: int
    num_scale_params: int | None
    default_identity_value: float
    model: Any = None
    transfer: SomaIdentityTransfer | None = None


def load(model_type: str, spec: Any, transfer: SomaIdentityTransfer | None = None) -> IdentityBackend:
    if model_type == "soma":
        return IdentityBackend(
            model_type=model_type,
            identity_dim=spec.identity_dim,
            num_scale_params=spec.num_scale_params,
            default_identity_value=spec.default_identity_value,
        )

    backend = IDENTITY_BACKENDS[model_type]
    model, transfer = backend.prepare(transfer)
    return IdentityBackend(
        model_type=model_type,
        identity_dim=spec.identity_dim,
        num_scale_params=spec.num_scale_params,
        default_identity_value=spec.default_identity_value,
        model=model,
        transfer=transfer,
    )


def rest_shape(
    *,
    backend: IdentityBackend,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    vertex_map: Any,
    xp: Any,
) -> tuple[Float[Any, "B Vt 3"], Float[Any, "B Va 3"]]:
    model_type = backend.model_type
    module = IDENTITY_BACKENDS[model_type]
    source_shape = module.shape(
        identity_model=backend.model,
        identity=identity,
        scale_params=scale_params,
        num_scale_params=backend.num_scale_params,
        xp=xp,
    )
    rest_shape_full = _transfer_source_shape(source_shape, cast(SomaIdentityTransfer, backend.transfer), xp=xp)
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
