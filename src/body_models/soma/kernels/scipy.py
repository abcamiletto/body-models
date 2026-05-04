"""SciPy-optimized SOMA kernels for the NumPy backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float
from scipy import sparse

from . import base

Array = Any

fit_rigid_transform = base.fit_rigid_transform
prepare_identity_shape = base.prepare_identity_shape
resolve_identity_inputs = base.resolve_identity_inputs


@dataclass(frozen=True)
class SomaScipyCorrectives(base.SomaCorrectives):
    corrective_W2: Any


class SomaScipyData(base.SomaData):
    correctives: SomaScipyCorrectives

    def apply_pose_correctives(
        self,
        pose_rot_full: Float[Array, "B Jf 3 3"],
        use_tanh: bool,
        *,
        xp: Any,
    ) -> Float[Array, "B V 3"]:
        correctives = self.correctives
        batch_size = pose_rot_full.shape[0]
        x = correctives.corrective_bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
        x = np.asarray(x, copy=True)
        x[:, :, 0, 0] -= 1.0
        x[:, :, 1, 1] -= 1.0
        feat = x[:, :, :, :2].reshape(batch_size, -1)

        z = feat @ correctives.corrective_W1
        z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
        if use_tanh:
            z = xp.tanh(z)

        return xp.asarray(z @ correctives.corrective_W2).reshape(batch_size, self.mean_full.shape[0], 3)

    def linear_blend_skinning(
        self,
        xp,
        bind_shape: Float[Array, "B V 3"],
        skin_weights: Any,
        bone_transforms: Float[Array, "B J 4 4"],
    ) -> Float[Array, "B V 3"]:
        R = bone_transforms[..., :3, :3]
        t = bone_transforms[..., :3, 3]
        out = xp.empty_like(bind_shape)
        for batch_index in range(bind_shape.shape[0]):
            R_blend = xp.asarray(skin_weights @ R[batch_index].reshape(R.shape[1], 9)).reshape(-1, 3, 3)
            t_blend = xp.asarray(skin_weights @ t[batch_index])
            out[batch_index] = xp.einsum("vik,vk->vi", R_blend, bind_shape[batch_index]) + t_blend
        return out


def prepare_data(**data):
    corrective_W2 = sparse.csr_matrix(
        (data["corrective_W2_values"], (data["corrective_W2_rows"], data["corrective_W2_cols"])),
        shape=(data["corrective_W1"].shape[1], data["mean_full"].shape[0] * 3),
    )
    correctives = SomaScipyCorrectives(
        corrective_bindpose=data["corrective_bindpose"],
        corrective_W1=data["corrective_W1"],
        corrective_W2_rows=data["corrective_W2_rows"],
        corrective_W2_cols=data["corrective_W2_cols"],
        corrective_W2_values=data["corrective_W2_values"],
        corrective_W2=corrective_W2,
    )
    prepared_data = {**data, "skin_weights_active": sparse.csr_matrix(data["skin_weights_active"])}
    return SomaScipyData.from_kernel_data_and_correctives(prepared_data, correctives)
