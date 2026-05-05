"""Identity sources used by SOMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float, Int

from ... import common
from ...anny import core as anny_core
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


def linear_identity_shape(
    mean: Float[Any, "V 3"],
    shapedirs: Float[Any, "V 3 I"],
    identity: Float[Any, "B I"],
    *,
    xp: Any = None,
) -> Float[Any, "B V 3"]:
    if xp is None:
        xp = common.get_namespace(identity)
    identity_dim = identity.shape[1]
    return mean[None] + xp.einsum("bi,vci->bvc", identity, shapedirs[..., :identity_dim])


def mhr_identity_shape(
    model: Any,
    identity: Float[Any, "B I"],
    scale_params: Float[Any, "B K"] | None,
    num_scale_params: int,
    *,
    xp: Any = None,
) -> Float[Any, "B V 3"]:
    if xp is None:
        xp = common.get_namespace(identity)

    batch_size = identity.shape[0]
    if scale_params is None:
        scale_params = common.zeros_as(identity, shape=(batch_size, num_scale_params), xp=xp)
    zero_pose = common.zeros_as(identity, shape=(batch_size, model.pose_dim), xp=xp)
    zero_pose = common.set(zero_pose, (slice(None), slice(-num_scale_params, None)), scale_params, xp=xp)
    expression = common.zeros_as(identity, shape=(batch_size, model.EXPR_DIM), xp=xp)
    return model.forward_vertices(shape=identity, pose=zero_pose, expression=expression)


def anny_identity_shape(
    template_vertices: Float[Any, "V 3"],
    blendshapes: Float[Any, "S V 3"],
    phenotype_mask: Float[Any, "S P"],
    anchors: dict[str, Float[Any, "A"]],
    identity: Float[Any, "B 6"],
    *,
    xp: Any = None,
) -> Float[Any, "B V 3"]:
    if xp is None:
        xp = common.get_namespace(identity)
    return anny_core.identity_shape(
        template_vertices=template_vertices,
        blendshapes=blendshapes,
        phenotype_mask=phenotype_mask,
        anchors=anchors,
        identity=identity,
        xp=xp,
    )


def transfer_identity_rest_shape(
    source_shape: Float[Any, "B Vs 3"],
    source_tetrahedra: Int[Any, "Fs 4"],
    face_ids: Int[Any, "Vt"],
    bary_coords: Float[Any, "Vt 4"],
    unknown_ids: Int[Any, "U"],
    anchor_ids: Int[Any, "A"],
    solve_matrix: Float[Any, "U U"],
    anchor_matrix: Float[Any, "U A"],
    rhs_base: Float[Any, "U 3"],
    *,
    xp: Any = None,
) -> Float[Any, "B Vt 3"]:
    if xp is None:
        xp = common.get_namespace(source_shape)

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
    rest_shape = transfer_identity_rest_shape(
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
