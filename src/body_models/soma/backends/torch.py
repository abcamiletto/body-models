"""PyTorch SOMA backend."""

import torch

from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "PreparedSomaIdentity",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
linear_blend_skinning = core.linear_blend_skinning
PreparedSomaIdentity = core.PreparedSomaIdentity


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
    x = x.clone()
    x[:, :, 0, 0] -= 1.0
    x[:, :, 1, 1] -= 1.0
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype, device=feat.device))

    contrib = z[:, correctives.corrective_W2_rows] * correctives.corrective_W2_values[None]
    out_shape = batch_size, data.mean_full.shape[0] * 3
    out = torch.zeros(out_shape, dtype=z.dtype, device=z.device)
    index = xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)
    return out.scatter_add(1, index, contrib).reshape(batch_size, data.mean_full.shape[0], 3)
