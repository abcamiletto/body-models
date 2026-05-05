"""Identity sources used by SOMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int

from .. import core
from ...common import get_namespace


@dataclass(frozen=True)
class IdentityTransfer:
    source_vertices: Float[Any, "Vs 3"]
    source_tetrahedra: Int[Any, "Fs 4"]
    face_ids: Int[Any, "Vt"]
    bary_coords: Float[Any, "Vt 4"]
    unknown_ids: Int[Any, "U"]
    anchor_ids: Int[Any, "A"]
    solve_matrix: Float[Any, "U U"]
    anchor_matrix: Float[Any, "U A"]
    rhs_base: Float[Any, "U 3"]
    internal_to_source_rotation: Float[Any, "3 3"]
    internal_to_source_translation: Float[Any, "3"]
    source_to_soma_rotation: Float[Any, "3 3"]
    source_scale: float
    output_scale: float


def linear_identity_shape(
    mean: Float[Any, "V 3"],
    shapedirs: Float[Any, "V 3 I"],
    identity: Float[Any, "B I"],
    *,
    xp: Any = None,
) -> Float[Any, "B V 3"]:
    if xp is None:
        xp = get_namespace(identity)
    identity_dim = identity.shape[1]
    return mean[None] + xp.einsum("bi,vci->bvc", identity, shapedirs[..., :identity_dim])


def transfer_shape(
    source_shape: Float[Any, "B Vs 3"],
    transfer: IdentityTransfer,
    *,
    vertex_map: Any,
    xp: Any,
) -> tuple[Float[Any, "B Vt 3"], Float[Any, "B Va 3"]]:
    rest_shape = core.apply_rigid_transform(
        source_shape,
        rotation=transfer.internal_to_source_rotation,
        translation=transfer.internal_to_source_translation,
        xp=xp,
    )
    rest_shape = rest_shape * transfer.source_scale
    rest_shape = core.transfer_identity_rest_shape(
        source_shape=rest_shape,
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
    rest_shape = core.apply_rigid_transform(
        rest_shape,
        rotation=transfer.source_to_soma_rotation,
        xp=xp,
    )
    rest_shape = rest_shape * transfer.output_scale
    rest_shape_active = rest_shape if vertex_map is None else rest_shape[:, vertex_map]
    return rest_shape, rest_shape_active
