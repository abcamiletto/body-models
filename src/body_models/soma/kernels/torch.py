"""PyTorch SOMA kernels."""

import torch

from . import base


class SomaTorchData(base.SomaData):
    def apply_pose_correctives(self, pose_rot_full, use_tanh: bool, *, xp=None):
        if xp is None:
            xp = torch

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
    data["topology"] = base.SomaTopology(
        parents_full=data.pop("parents_full"),
        parents_full_index=data.pop("parents_full_index"),
        joint_children_full=data.pop("joint_children_full"),
        joint_children_indices_full=data.pop("joint_children_indices_full"),
        skinned_vertex_indices_full=data.pop("skinned_vertex_indices_full"),
        skinned_vertex_indices_full_index=data.pop("skinned_vertex_indices_full_index"),
        kinematic_fronts_full=data.pop("kinematic_fronts_full"),
    )
    data["correctives"] = base.SomaCorrectives(
        corrective_bindpose=data.pop("corrective_bindpose"),
        corrective_W1=data.pop("corrective_W1"),
        corrective_W2_rows=data.pop("corrective_W2_rows"),
        corrective_W2_cols=data.pop("corrective_W2_cols"),
        corrective_W2_values=data.pop("corrective_W2_values"),
        corrective_W2=data.pop("corrective_W2"),
    )
    return SomaTorchData(**data)


ops = base.SomaOps()
