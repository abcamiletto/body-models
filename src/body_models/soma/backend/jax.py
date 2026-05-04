"""JAX SOMA backend."""

from dataclasses import replace

import jax.numpy as jnp

from ..io import SomaWeights
from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "prepare_data",
    "prepare_identity",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
linear_blend_skinning = core.linear_blend_skinning
prepare_identity = core.prepare_identity


def prepare_data(weights: SomaWeights) -> SomaWeights:
    topology = replace(
        weights.topology,
        joint_children_indices_full=jnp.asarray(weights.topology.joint_children_indices_full),
        skinned_vertex_indices_full_index=jnp.asarray(weights.topology.skinned_vertex_indices_full_index),
    )
    correctives = replace(
        weights.correctives,
        corrective_bindpose=jnp.asarray(weights.correctives.corrective_bindpose),
        corrective_W1=jnp.asarray(weights.correctives.corrective_W1),
        corrective_W2_rows=jnp.asarray(weights.correctives.corrective_W2_rows),
        corrective_W2_cols=jnp.asarray(weights.correctives.corrective_W2_cols),
        corrective_W2_values=jnp.asarray(weights.correctives.corrective_W2_values),
    )
    return replace(
        weights,
        mean_full=jnp.asarray(weights.mean_full),
        mean_active=jnp.asarray(weights.mean_active),
        shapedirs_full=jnp.asarray(weights.shapedirs_full),
        shapedirs_active=jnp.asarray(weights.shapedirs_active),
        eigenvalues=jnp.asarray(weights.eigenvalues),
        bind_shape_full=jnp.asarray(weights.bind_shape_full),
        bind_pose_world=jnp.asarray(weights.bind_pose_world),
        bind_pose_local=jnp.asarray(weights.bind_pose_local),
        t_pose_world=jnp.asarray(weights.t_pose_world),
        t_pose_local=jnp.asarray(weights.t_pose_local),
        joint_regressor=jnp.asarray(weights.joint_regressor),
        skin_weights_full=jnp.asarray(weights.skin_weights_full),
        skin_weights_active=jnp.asarray(weights.skin_weights_active),
        faces=jnp.asarray(weights.faces),
        vertex_map=None if weights.vertex_map is None else jnp.asarray(weights.vertex_map),
        facial_inner_vertices=jnp.asarray(weights.facial_inner_vertices),
        topology=topology,
        correctives=correctives,
    )


def forward_vertices(*args, **kwargs):
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def apply_pose_correctives(data, pose_rot_full, *, xp):
    correctives = data.correctives
    batch_size = pose_rot_full.shape[0]
    x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = x.at[:, :, 0, 0].add(-1.0)
    x = x.at[:, :, 1, 1].add(-1.0)
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out = xp.zeros((batch_size, data.mean_full.shape[0] * 3), dtype=z.dtype)
    out = out.at[:, correctives.corrective_W2_cols].add(contrib)
    return out.reshape(batch_size, data.mean_full.shape[0], 3)
