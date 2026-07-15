"""PyTorch SOMA backend."""

import torch

from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "SomaIdentity",
    "SomaPreparedPose",
    "prepare_data",
    "prepare_pose",
    "prepare_identity_from_rest_shape",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
SomaIdentity = core.SomaIdentity
SomaPreparedPose = core.SomaPreparedPose


def prepare_data(data):
    data.topology.register_buffer(
        "parents_full_index",
        torch.tensor(data.topology.parents_full),
        persistent=False,
    )
    if data.public is not None:
        data.public.topology.register_buffer(
            "parents_full_index",
            torch.tensor(data.public.topology.parents_full),
            persistent=False,
        )

    correctives = data.correctives
    output_size = data.mean_full.shape[0] * 3
    hidden_size = correctives.corrective_W1.shape[1]
    indices = torch.stack((correctives.corrective_W2_cols, correctives.corrective_W2_rows))
    transpose = torch.sparse_coo_tensor(
        indices,
        correctives.corrective_W2_values,
        (output_size, hidden_size),
    ).coalesce()
    correctives.register_buffer("corrective_W2_transpose", transpose, persistent=False)
    return data


def prepare_identity_from_rest_shape(*args, **kwargs):
    return core.prepare_identity_from_rest_shape(
        *args,
        **kwargs,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def forward_vertices(*args, **kwargs):
    return core._forward_vertices_with(
        *args,
        **kwargs,
        linear_blend_skinning_fn=linear_blend_skinning,
    )


def prepare_pose(*args, **kwargs):
    return core.prepare_pose(*args, **kwargs, apply_pose_correctives_fn=apply_pose_correctives)


def linear_blend_skinning(xp, bind_shape, skin_weights, skinning_transforms):
    """Blend affine transforms before applying them to the vertices."""
    affine = torch.einsum("vj,...jck->...vck", skin_weights, skinning_transforms[..., :3, :])
    return torch.einsum("...vck,...vk->...vc", affine[..., :3], bind_shape) + affine[..., 3]


def apply_pose_correctives(data, pose_rot_full, *, xp):
    correctives = data.correctives
    batch_shape = pose_rot_full.shape[:-3]
    x = correctives.corrective_bindpose.swapaxes(-2, -1) @ pose_rot_full
    x = x.clone()
    x[..., :, 0, 0] -= 1.0
    x[..., :, 1, 1] -= 1.0
    feat = x[..., :, :, :2].reshape(*batch_shape, -1)

    z = feat @ correctives.corrective_W1
    z = torch.relu(z)

    # Dynamo cannot trace sparse tensors; Inductor can fuse this equivalent form.
    if torch.compiler.is_compiling():
        contrib = z[..., correctives.corrective_W2_rows] * correctives.corrective_W2_values
        output_size = data.mean_full.shape[0] * 3
        out = torch.zeros((*batch_shape, output_size), dtype=z.dtype, device=z.device)
        indices = torch.broadcast_to(correctives.corrective_W2_cols, contrib.shape)
        out = out.scatter_add(-1, indices, contrib)
        return out.reshape(*batch_shape, data.mean_full.shape[0], 3)

    flat = torch.sparse.mm(correctives.corrective_W2_transpose, z.reshape(-1, z.shape[-1]).T).T
    return flat.reshape(*batch_shape, data.mean_full.shape[0], 3)
