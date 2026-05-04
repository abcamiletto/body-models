"""PyTorch SOMA backend."""

import torch

from . import core

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
prepare_identity_shape = core.prepare_identity_shape
prepare_identity_state = core.prepare_identity_state
resolve_identity_inputs = core.resolve_identity_inputs
prepare_data = core.prepare_data


def forward_vertices(*args, **kwargs):
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=core.linear_blend_skinning,
    )


def apply_pose_correctives(data, pose_rot_full, use_tanh: bool, *, xp):
    correctives = data.correctives
    batch_size = pose_rot_full.shape[0]
    x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = x.clone()
    x[:, :, 0, 0] -= 1.0
    x[:, :, 1, 1] -= 1.0
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype, device=feat.device))
    if use_tanh:
        z = xp.tanh(z)

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out = torch.zeros((batch_size, data.mean_full.shape[0] * 3), dtype=z.dtype, device=z.device)
    index = xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)
    return out.scatter_add(1, index, contrib).reshape(batch_size, data.mean_full.shape[0], 3)
