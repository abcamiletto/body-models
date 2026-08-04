"""Vertex mappings evaluated directly through linear blend skinning."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

import numpy as np
from jaxtyping import Float

from body_models import _state as state
from body_models._common import deformation, sparse
from body_models._runtime import ArrayRuntime

Array = Any


class PointRegressor(TypedDict):
    """A vertex mapping projected through a model's static deformation state."""

    weight_sums: Float[Array, "K"]
    skinning_weights: Float[Array, "K J"]
    vertex_projection: sparse.SparseLinear
    corrective_basis: deformation.CorrectiveBasis | None
    template: NotRequired[Float[Array, "K J 3"]]
    identity_directions: NotRequired[Float[Array, "K J 3 C"]]


def prepare_point_regressor(
    mapping: Float[Array, "K V"],
    skinning_weights: Float[Array, "V J"],
    corrective_basis: deformation.CorrectiveBasis | None,
    *,
    runtime: ArrayRuntime,
) -> PointRegressor:
    """Project model-static skinning and corrective data through a mapping."""
    xp = runtime.xp
    point_skinning_weights = mapping @ skinning_weights
    vertex_projection = _prepare_vertex_projection(mapping, skinning_weights, runtime=runtime)
    regressor: PointRegressor = {
        "weight_sums": xp.sum(mapping, axis=-1),
        "skinning_weights": point_skinning_weights,
        "vertex_projection": vertex_projection,
        "corrective_basis": _project_corrective_basis(
            point_skinning_weights,
            vertex_projection,
            corrective_basis,
            runtime=runtime,
        ),
    }
    if runtime.name == "jax":
        state.register_jax_state(regressor)
    return regressor


def project_vertex_values(
    regressor: PointRegressor,
    values: Float[Array, "V *dims"],
    *,
    xp: Any,
) -> Float[Array, "K J *dims"]:
    """Apply a regressor's sparse vertex projection to vertex-first values."""
    return _project_vertex_values(
        regressor["vertex_projection"],
        regressor["skinning_weights"].shape,
        values,
        xp=xp,
    )


def _project_vertex_values(
    vertex_projection: sparse.SparseLinear,
    point_shape: tuple[int, int],
    values: Float[Array, "V *dims"],
    *,
    xp: Any,
) -> Float[Array, "K J *dims"]:
    values = xp.moveaxis(values, 0, -1)
    projected = vertex_projection(values)
    num_points, num_joints = point_shape
    projected = projected.reshape(*values.shape[:-1], num_points, num_joints)
    return xp.moveaxis(projected, (-2, -1), (0, 1))


def project_rest_points(
    regressor: PointRegressor,
    rest_vertices: Float[Array, "*batch V 3"],
    *,
    xp: Any,
) -> Float[Array, "*batch K J 3"]:
    """Project identity-dependent rest vertices into point-joint space."""
    values = xp.moveaxis(rest_vertices, -2, -1)
    projected = regressor["vertex_projection"](values)
    num_points, num_joints = regressor["skinning_weights"].shape
    projected = projected.reshape(*values.shape[:-1], num_points, num_joints)
    return xp.moveaxis(projected, -3, -1)


def regress_points(
    regressor: PointRegressor,
    rest_points: Float[Array, "*batch K J 3"],
    pose: deformation.SkinningPose,
    *,
    xp: Any,
) -> Float[Array, "*batch K 3"]:
    """Skin projected rest points with prepared pose state."""
    points = rest_points
    coefficients = pose.get("pose_coefficients")
    if coefficients is not None:
        basis = regressor["corrective_basis"]
        if basis is None:
            raise RuntimeError("Prepared pose has corrective coefficients, but the point regressor has no basis.")
        offsets = basis.apply(coefficients)
        points = points + offsets.reshape(*offsets.shape[:-2], *rest_points.shape[-3:])

    transforms = pose["skinning_transforms"]
    positions = xp.einsum("...jcd,...kjd->...kc", transforms[..., :3, :3], points)
    translations = xp.einsum(
        "kj,...jc->...kc",
        regressor["skinning_weights"],
        transforms[..., :3, 3],
    )
    return positions + translations


def _project_corrective_basis(
    point_skinning_weights: Float[Array, "K J"],
    vertex_projection: sparse.SparseLinear,
    basis: deformation.CorrectiveBasis | None,
    *,
    runtime: ArrayRuntime,
) -> deformation.CorrectiveBasis | None:
    if basis is None:
        return None
    if isinstance(basis, deformation.DenseCorrectiveBasis):
        values = basis.values.reshape(basis.coefficient_dim, basis.num_vertices, 3)
        directions = runtime.xp.moveaxis(values, 0, -1)
        projected = _project_vertex_values(
            vertex_projection,
            point_skinning_weights.shape,
            directions,
            xp=runtime.xp,
        )
        projected = runtime.xp.moveaxis(projected, -1, 0).reshape(basis.coefficient_dim, -1)
        return deformation.DenseCorrectiveBasis(projected)
    return _project_sparse_corrective_basis(vertex_projection, basis, runtime=runtime)


def _project_sparse_corrective_basis(
    vertex_projection: sparse.SparseLinear,
    basis: deformation.SparseCorrectiveBasis,
    *,
    runtime: ArrayRuntime,
) -> deformation.SparseCorrectiveBasis:
    from scipy import sparse as scipy_sparse

    source = basis.to_coo()
    projection = vertex_projection.to_coo()
    source_matrix = scipy_sparse.coo_matrix(
        (
            runtime.to_numpy(source.values),
            (
                runtime.to_numpy(source.row_indices),
                runtime.to_numpy(source.column_indices),
            ),
        ),
        shape=source.shape,
    )
    coordinates = np.arange(3)
    projection_rows = runtime.to_numpy(projection.row_indices)[:, None] * 3 + coordinates
    projection_columns = runtime.to_numpy(projection.column_indices)[:, None] * 3 + coordinates
    projection_values = np.repeat(runtime.to_numpy(projection.values), 3)
    projection_matrix = scipy_sparse.coo_matrix(
        (
            projection_values,
            (projection_rows.reshape(-1), projection_columns.reshape(-1)),
        ),
        shape=(projection.shape[0] * 3, projection.shape[1] * 3),
    )
    projected = (source_matrix @ projection_matrix).tocoo()

    projected_matrix = sparse.SparseMatrix(
        row_indices=runtime.asarray(
            projected.row,
            like=source.row_indices,
            dtype=source.row_indices.dtype,
        ),
        column_indices=runtime.asarray(
            projected.col,
            like=source.column_indices,
            dtype=source.column_indices.dtype,
        ),
        values=runtime.asarray(projected.data, like=source.values),
        shape=projected.shape,
    )
    materialized = runtime._materialize(projected_matrix)
    return deformation.SparseCorrectiveBasis(materialized)


def _prepare_vertex_projection(
    mapping: Float[Array, "K V"],
    skinning_weights: Float[Array, "V J"],
    *,
    runtime: ArrayRuntime,
) -> sparse.SparseLinear:
    mapping_array = mapping
    mapping = runtime.to_numpy(mapping_array)
    weights_numpy = runtime.to_numpy(skinning_weights)
    num_vertices, num_joints = weights_numpy.shape

    rows = []
    columns = []
    values = []
    for vertex_index in range(num_vertices):
        point_indices = np.flatnonzero(mapping[:, vertex_index])
        joint_indices = np.flatnonzero(weights_numpy[vertex_index])
        if point_indices.size == 0 or joint_indices.size == 0:
            continue
        rows.append(np.full(point_indices.size * joint_indices.size, vertex_index))
        columns.append((point_indices[:, None] * num_joints + joint_indices).reshape(-1))
        weights = mapping[point_indices, vertex_index, None] * weights_numpy[vertex_index, joint_indices]
        values.append(weights.reshape(-1))

    row_indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    column_indices = np.concatenate(columns) if columns else np.empty(0, dtype=np.int32)
    projection_values = np.concatenate(values) if values else np.empty(0, dtype=mapping.dtype)
    matrix = sparse.SparseMatrix(
        row_indices=runtime.asarray(row_indices, like=mapping_array, dtype=runtime.xp.int32),
        column_indices=runtime.asarray(column_indices, like=mapping_array, dtype=runtime.xp.int32),
        values=runtime.asarray(projection_values, like=mapping_array),
        shape=(num_vertices, mapping.shape[0] * num_joints),
    )
    return runtime._materialize(matrix)


__all__ = ["PointRegressor"]
