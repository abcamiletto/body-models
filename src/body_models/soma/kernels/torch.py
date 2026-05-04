"""PyTorch SOMA kernels."""

import torch

from . import base

fit_rigid_transform = base.fit_rigid_transform
prepare_identity_shape = base.prepare_identity_shape
resolve_identity_inputs = base.resolve_identity_inputs


class SomaTorchData(base.SomaData):
    def apply_pose_correctives(self, pose_rot_full, use_tanh: bool, *, xp):
        correctives = self.correctives
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
        out = torch.zeros((batch_size, self.mean_full.shape[0] * 3), dtype=z.dtype, device=z.device)
        index = xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)
        return out.scatter_add(1, index, contrib).reshape(batch_size, self.mean_full.shape[0], 3)


def prepare_data(**data):
    return SomaTorchData.from_kernel_data(data)
