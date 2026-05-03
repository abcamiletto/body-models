"""JAX SOMA kernels."""

from . import base


class SomaJaxData(base.SomaData):
    def apply_pose_correctives(self, pose_rot_full, use_tanh: bool, *, xp):
        correctives = self.correctives
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
        out = xp.zeros((batch_size, self.mean_full.shape[0] * 3), dtype=z.dtype)
        out = out.at[:, correctives.corrective_W2_cols].add(contrib)
        return out.reshape(batch_size, self.mean_full.shape[0], 3)


def prepare_data(**data):
    return SomaJaxData.from_kernel_data(data)


ops = base.SomaOps()
