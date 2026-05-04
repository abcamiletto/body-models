"""JAX SOMA backend."""

from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "prepare_data",
    "prepare_identity_shape",
    "prepare_identity_state",
    "resolve_identity_inputs",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
linear_blend_skinning = core.linear_blend_skinning
prepare_data = core.prepare_data
prepare_identity_shape = core.prepare_identity_shape
prepare_identity_state = core.prepare_identity_state
resolve_identity_inputs = core.resolve_identity_inputs


def forward_vertices(*args, **kwargs):
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def apply_pose_correctives(data, pose_rot_full, use_tanh: bool, *, xp):
    correctives = data.correctives
    batch_size = pose_rot_full.shape[0]
    x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = x.at[:, :, 0, 0].add(-1.0)
    x = x.at[:, :, 1, 1].add(-1.0)
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
    if use_tanh:
        z = xp.tanh(z)

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out = xp.zeros((batch_size, data.mean_full.shape[0] * 3), dtype=z.dtype)
    out = out.at[:, correctives.corrective_W2_cols].add(contrib)
    return out.reshape(batch_size, data.mean_full.shape[0], 3)
