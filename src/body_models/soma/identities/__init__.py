"""Identity sources used by SOMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float

from .. import core


@dataclass(frozen=True)
class IdentityTransfer:
    source_vertices: Any
    source_tetrahedra: Any
    face_ids: Any
    bary_coords: Any
    unknown_ids: Any
    anchor_ids: Any
    solve_matrix: Any
    anchor_matrix: Any
    rhs_base: Any
    internal_to_source_rotation: Any
    internal_to_source_translation: Any
    source_to_soma_rotation: Any
    source_scale: float
    output_scale: float


def transfer_from_data(transfer: Any) -> IdentityTransfer:
    return IdentityTransfer(**transfer.__dict__)


def with_transforms(
    transfer: IdentityTransfer,
    *,
    internal_to_source_rotation: Any,
    internal_to_source_translation: Any,
    source_to_soma_rotation: Any,
) -> IdentityTransfer:
    return IdentityTransfer(
        source_vertices=transfer.source_vertices,
        source_tetrahedra=transfer.source_tetrahedra,
        face_ids=transfer.face_ids,
        bary_coords=transfer.bary_coords,
        unknown_ids=transfer.unknown_ids,
        anchor_ids=transfer.anchor_ids,
        solve_matrix=transfer.solve_matrix,
        anchor_matrix=transfer.anchor_matrix,
        rhs_base=transfer.rhs_base,
        internal_to_source_rotation=internal_to_source_rotation,
        internal_to_source_translation=internal_to_source_translation,
        source_to_soma_rotation=source_to_soma_rotation,
        source_scale=transfer.source_scale,
        output_scale=transfer.output_scale,
    )


def source_shape(
    model_type: str,
    *,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int | None,
    mhr_model: Any = None,
    anny_model: Any = None,
    linear_model: Any = None,
    xp: Any,
) -> Float[Any, "B V 3"]:
    if model_type == "mhr":
        return core.mhr_identity_shape(
            model=mhr_model,
            identity=identity,
            scale_params=scale_params,
            num_scale_params=_require_scale_params(num_scale_params),
            xp=xp,
        )
    if model_type == "anny":
        return core.anny_identity_shape(
            template_vertices=_array(anny_model.template_vertices),
            blendshapes=_array(anny_model.blendshapes),
            phenotype_mask=_array(anny_model.phenotype_mask),
            anchors=_anny_anchors(anny_model),
            identity=identity,
            xp=xp,
        )
    return core.linear_identity_shape(
        mean=_array(linear_model.v_template_full),
        shapedirs=_array(linear_model.shapedirs_full),
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


def _require_scale_params(num_scale_params: int | None) -> int:
    if num_scale_params is None:
        raise ValueError("SOMA model_type='mhr' requires scale parameters.")
    return num_scale_params


def _array(value: Any) -> Any:
    return value[...] if hasattr(value, "__getitem__") else value


def _anny_anchors(model: Any) -> Any:
    if hasattr(model, "_get_anchors_dict"):
        return model._get_anchors_dict()
    return model._anchors
