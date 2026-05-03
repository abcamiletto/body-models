"""SciPy-optimized SOMA kernels for the NumPy backend."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from scipy import sparse

from . import base

Array = Any


def apply_pose_correctives(
    pose_rot_full: Float[Array, "B Jf 3 3"],
    bindpose: Float[Array, "Jf 3 3"],
    W1: Float[Array, "D K"],
    W2_rows: Int[Array, "NNZ"],
    W2_cols: Int[Array, "NNZ"],
    W2_values: Float[Array, "NNZ"],
    num_vertices: int,
    use_tanh: bool,
    *,
    xp: Any = None,
) -> Float[Array, "B V 3"]:
    if xp is None:
        xp = np

    batch_size = pose_rot_full.shape[0]
    x = bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = np.asarray(x, copy=True)
    x[:, :, 0, 0] -= 1.0
    x[:, :, 1, 1] -= 1.0
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
    if use_tanh:
        z = xp.tanh(z)

    W2 = sparse.csr_matrix((W2_values, (W2_rows, W2_cols)), shape=(W1.shape[1], num_vertices * 3))
    return xp.asarray(z @ W2).reshape(batch_size, num_vertices, 3)


def linear_blend_skinning(
    xp,
    bind_shape: Float[Array, "B V 3"],
    skin_weights: Float[Array, "V J"],
    bone_transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B V 3"]:
    skin_weights = sparse.csr_matrix(skin_weights)
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    out = xp.empty_like(bind_shape)
    for batch_index in range(bind_shape.shape[0]):
        R_blend = xp.asarray(skin_weights @ R[batch_index].reshape(R.shape[1], 9)).reshape(-1, 3, 3)
        t_blend = xp.asarray(skin_weights @ t[batch_index])
        out[batch_index] = xp.einsum("vik,vk->vi", R_blend, bind_shape[batch_index]) + t_blend
    return out


ops = replace(
    base.SomaOps(),
    apply_pose_correctives=apply_pose_correctives,
    linear_blend_skinning=linear_blend_skinning,
)
