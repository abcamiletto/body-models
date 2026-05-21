"""SciPy-optimized SOMA backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from jaxtyping import Float
from scipy import sparse

from ..io import SomaCorrectives
from . import core

Array = Any

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "SomaIdentity",
    "prepare_data",
    "prepare_pose",
    "prepare_identity_from_rest_shape",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
SomaIdentity = core.SomaIdentity
SomaPreparedPose = core.SomaPreparedPose
prepare_pose = core.prepare_pose


def prepare_identity_from_rest_shape(*args, **kwargs):
    return core.prepare_identity_from_rest_shape(
        *args,
        **kwargs,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


@dataclass(frozen=True)
class SomaScipyCorrectives(SomaCorrectives):
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
    *,
    xp: Any,
) -> Float[Array, "B V 3"]:
    correctives = data.correctives
    batch_shape = pose_rot_full.shape[:-3]
    x = correctives.corrective_bindpose.swapaxes(-2, -1) @ pose_rot_full
    x = np.asarray(x, copy=True)
    x[..., :, 0, 0] -= 1.0
    x[..., :, 1, 1] -= 1.0
    feat = x[..., :, :, :2].reshape(*batch_shape, -1)

    z = feat @ correctives.corrective_W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))

    return xp.asarray(z @ correctives.corrective_W2).reshape(*batch_shape, data.mean_full.shape[0], 3)


def linear_blend_skinning(
    xp,
    bind_shape: Float[Array, "B V 3"],
    skin_weights: Any,
    bone_transforms: Float[Array, "B J 4 4"],
) -> Float[Array, "B V 3"]:
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    out = xp.empty_like(bind_shape)
    for batch_index in np.ndindex(bind_shape.shape[:-2]):
        R_blend = xp.asarray(skin_weights @ R[batch_index].reshape(R.shape[-3], 9)).reshape(-1, 3, 3)
        t_blend = xp.asarray(skin_weights @ t[batch_index])
        out[batch_index] = xp.einsum("vik,vk->vi", R_blend, bind_shape[batch_index]) + t_blend
    return out


def prepare_data(weights):
    source = weights.correctives
    values = source.corrective_W2_values
    indices = source.corrective_W2_rows, source.corrective_W2_cols
    shape = source.corrective_W1.shape[1], weights.mean_full.shape[0] * 3
    corrective_W2 = sparse.csr_matrix((values, indices), shape=shape)
    correctives = SomaScipyCorrectives(
        corrective_bindpose=source.corrective_bindpose,
        corrective_W1=source.corrective_W1,
        corrective_W2_rows=source.corrective_W2_rows,
        corrective_W2_cols=source.corrective_W2_cols,
        corrective_W2_values=source.corrective_W2_values,
        corrective_W2=corrective_W2,
    )
    prepared = core.prepare_data(weights)
    return replace(
        prepared,
        skin_weights_active=sparse.csr_matrix(weights.skin_weights_active),
        correctives=correctives,
    )
