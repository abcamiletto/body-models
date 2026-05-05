"""Identity sources used by SOMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int

from .. import core


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


def mhr_source_shape(
    *,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int,
    model: Any,
    xp: Any,
) -> Float[Any, "B V 3"]:
    return core.mhr_identity_shape(
        model=model,
        identity=identity,
        scale_params=scale_params,
        num_scale_params=num_scale_params,
        xp=xp,
    )


def anny_source_shape(
    *,
    template_vertices: Float[Any, "V 3"],
    blendshapes: Float[Any, "P V 3"],
    phenotype_mask: Float[Any, "P"],
    anchors: Any,
    identity: Float[Any, "B I"],
    xp: Any,
) -> Float[Any, "B V 3"]:
    return core.anny_identity_shape(
        template_vertices=template_vertices,
        blendshapes=blendshapes,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        identity=identity,
        xp=xp,
    )


def linear_source_shape(
    *,
    mean: Float[Any, "V 3"],
    shapedirs: Float[Any, "I V 3"],
    identity: Float[Any, "B I"],
    xp: Any,
) -> Float[Any, "B V 3"]:
    return core.linear_identity_shape(
        mean=mean,
        shapedirs=shapedirs,
        identity=identity,
        xp=xp,
    )


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
