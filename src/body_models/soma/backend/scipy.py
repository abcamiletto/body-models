"""SciPy-optimized SOMA backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float
from scipy import sparse

from . import core

Array = Any

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
prepare_identity_shape = core.prepare_identity_shape
prepare_identity_state = core.prepare_identity_state
resolve_identity_inputs = core.resolve_identity_inputs


@dataclass(frozen=True)
class SomaScipyCorrectives(core.SomaCorrectives):
    corrective_W2: Any


def forward_vertices(*args, **kwargs):
    return core._forward_vertices_with(
        *args,
        **kwargs,
        apply_pose_correctives_fn=apply_pose_correctives,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def apply_pose_correctives(
    data: Any,
    pose_rot_full: Float[Array, "B Jf 3 3"],
    use_tanh: bool,
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    correctives = data.correctives
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

    return xp.asarray(z @ correctives.corrective_W2).reshape(batch_size, data.mean_full.shape[0], 3)


def linear_blend_skinning(
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
    return core.prepare_data_with_correctives(correctives=correctives, **prepared_data)
