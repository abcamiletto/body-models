"""JAX SOMA kernels."""

import jax.numpy as jnp

from . import base


class SomaJaxData(base.SomaData):
    def apply_pose_correctives(self, pose_rot_full, use_tanh: bool, *, xp=None):
        if xp is None:
            xp = jnp

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
    return SomaJaxData(**data)


ops = base.SomaOps()
