"""PyTorch SOMA backend."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .. import identities
from ..io import SomaIdentityTransfer, SomaWeights
from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "prepare_data",
    "prepare_identity",
    "prepare_identity_backend",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
linear_blend_skinning = core.linear_blend_skinning
prepare_identity = core.prepare_identity


class SomaTorchCorrectives(nn.Module):
    corrective_bindpose: Float[torch.Tensor, "Jf 3 3"]
    corrective_W1: Float[torch.Tensor, "D K"]
    corrective_W2_rows: Int[torch.Tensor, "NNZ"]
    corrective_W2_cols: Int[torch.Tensor, "NNZ"]
    corrective_W2_values: Float[torch.Tensor, "NNZ"]

    def __init__(self, correctives):
        super().__init__()
        self.register_buffer("corrective_bindpose", torch.as_tensor(correctives.corrective_bindpose))
        self.register_buffer("corrective_W1", torch.as_tensor(correctives.corrective_W1))
        self.register_buffer("corrective_W2_rows", torch.as_tensor(correctives.corrective_W2_rows, dtype=torch.int64))
        self.register_buffer("corrective_W2_cols", torch.as_tensor(correctives.corrective_W2_cols, dtype=torch.int64))
        self.register_buffer("corrective_W2_values", torch.as_tensor(correctives.corrective_W2_values))


class SomaTorchTopology(nn.Module):
    joint_children_indices_full: Int[torch.Tensor, "Jf C"]
    skinned_vertex_indices_full_index: Int[torch.Tensor, "Jf K"]

    def __init__(self, topology):
        super().__init__()
        self.parents_full = topology.parents_full
        self.joint_children_full = topology.joint_children_full
        self.skinned_vertex_indices_full = topology.skinned_vertex_indices_full
        self.kinematic_fronts_full = topology.kinematic_fronts_full
        self.register_buffer("joint_children_indices_full", torch.as_tensor(topology.joint_children_indices_full))
        self.register_buffer(
            "skinned_vertex_indices_full_index",
            torch.as_tensor(topology.skinned_vertex_indices_full_index),
        )


class SomaTorchWeights(nn.Module):
    mean_full: Float[torch.Tensor, "Vf 3"]
    mean_active: Float[torch.Tensor, "Va 3"]
    shapedirs_full: Float[torch.Tensor, "S Vf 3"]
    shapedirs_active: Float[torch.Tensor, "S Va 3"]
    eigenvalues: Float[torch.Tensor, "S"]
    bind_shape_full: Float[torch.Tensor, "Vf 3"]
    bind_pose_world: Float[torch.Tensor, "Jf 4 4"]
    bind_pose_local: Float[torch.Tensor, "Jf 4 4"]
    t_pose_world: Float[torch.Tensor, "Jf 4 4"]
    t_pose_local: Float[torch.Tensor, "Jf 4 4"]
    joint_regressor: Float[torch.Tensor, "Jf Vf"]
    skin_weights_full: Float[torch.Tensor, "Vf Jf"]
    skin_weights_active: Float[torch.Tensor, "Va Jf"]
    faces: Int[torch.Tensor, "F 3"]
    facial_inner_vertices: Int[torch.Tensor, "Va"]
    vertex_map: Int[torch.Tensor, "Va"] | None
    topology: SomaTorchTopology
    correctives: SomaTorchCorrectives

    def __init__(self, weights):
        super().__init__()
        self.topology = SomaTorchTopology(weights.topology)
        self.correctives = SomaTorchCorrectives(weights.correctives)
        self.joint_names_full = weights.joint_names_full

        self.register_buffer("mean_full", torch.as_tensor(weights.mean_full))
        self.register_buffer("mean_active", torch.as_tensor(weights.mean_active))
        self.register_buffer("shapedirs_full", torch.as_tensor(weights.shapedirs_full))
        self.register_buffer("shapedirs_active", torch.as_tensor(weights.shapedirs_active))
        self.register_buffer("eigenvalues", torch.as_tensor(weights.eigenvalues))
        self.register_buffer("bind_shape_full", torch.as_tensor(weights.bind_shape_full))
        self.register_buffer("bind_pose_world", torch.as_tensor(weights.bind_pose_world))
        self.register_buffer("bind_pose_local", torch.as_tensor(weights.bind_pose_local))
        self.register_buffer("t_pose_world", torch.as_tensor(weights.t_pose_world))
        self.register_buffer("t_pose_local", torch.as_tensor(weights.t_pose_local))
        self.register_buffer("joint_regressor", torch.as_tensor(weights.joint_regressor))
        self.register_buffer("skin_weights_full", torch.as_tensor(weights.skin_weights_full))
        self.register_buffer("skin_weights_active", torch.as_tensor(weights.skin_weights_active))
        self.register_buffer("faces", torch.as_tensor(weights.faces, dtype=torch.int64))
        self.register_buffer("facial_inner_vertices", torch.as_tensor(weights.facial_inner_vertices, dtype=torch.int64))
        self.register_buffer(
            "vertex_map",
            None if weights.vertex_map is None else torch.as_tensor(weights.vertex_map, dtype=torch.int64),
        )


class SomaTorchIdentityTransfer(nn.Module):
    source_vertices: Float[torch.Tensor, "Vs 3"]
    source_tetrahedra: Int[torch.Tensor, "Fs 4"]
    face_ids: Int[torch.Tensor, "Vt"]
    bary_coords: Float[torch.Tensor, "Vt 4"]
    unknown_ids: Int[torch.Tensor, "U"]
    anchor_ids: Int[torch.Tensor, "A"]
    solve_matrix: Float[torch.Tensor, "U U"]
    anchor_matrix: Float[torch.Tensor, "U A"]
    rhs_base: Float[torch.Tensor, "U 3"]
    internal_to_source_rotation: Float[torch.Tensor, "3 3"]
    internal_to_source_translation: Float[torch.Tensor, "3"]
    source_to_soma_rotation: Float[torch.Tensor, "3 3"]

    def __init__(self, identity_transfer: SomaIdentityTransfer):
        super().__init__()
        self.source_scale = identity_transfer.source_scale
        self.output_scale = identity_transfer.output_scale
        self.register_buffer("source_vertices", torch.as_tensor(identity_transfer.source_vertices))
        self.register_buffer(
            "source_tetrahedra", torch.as_tensor(identity_transfer.source_tetrahedra, dtype=torch.int64)
        )
        self.register_buffer("face_ids", torch.as_tensor(identity_transfer.face_ids, dtype=torch.int64))
        self.register_buffer("bary_coords", torch.as_tensor(identity_transfer.bary_coords))
        self.register_buffer("unknown_ids", torch.as_tensor(identity_transfer.unknown_ids, dtype=torch.int64))
        self.register_buffer("anchor_ids", torch.as_tensor(identity_transfer.anchor_ids, dtype=torch.int64))
        self.register_buffer("solve_matrix", torch.as_tensor(identity_transfer.solve_matrix))
        self.register_buffer("anchor_matrix", torch.as_tensor(identity_transfer.anchor_matrix))
        self.register_buffer("rhs_base", torch.as_tensor(identity_transfer.rhs_base))
        self.register_buffer(
            "internal_to_source_rotation", torch.as_tensor(identity_transfer.internal_to_source_rotation)
        )
        self.register_buffer(
            "internal_to_source_translation",
            torch.as_tensor(identity_transfer.internal_to_source_translation),
        )
        self.register_buffer("source_to_soma_rotation", torch.as_tensor(identity_transfer.source_to_soma_rotation))


class SomaTorchIdentityBackend(nn.Module):
    def __init__(self, identity_backend: identities.TransferredIdentityBackend):
        super().__init__()
        self.model_type = identity_backend.model_type
        self.identity_dim = identity_backend.identity_dim
        self.num_scale_params = identity_backend.num_scale_params
        self.default_identity_value = identity_backend.default_identity_value
        self.model = identity_backend.model
        self.transfer = SomaTorchIdentityTransfer(identity_backend.transfer)


def prepare_data(weights: SomaWeights) -> SomaTorchWeights:
    return SomaTorchWeights(weights)


def prepare_identity_backend(identity_backend: identities.IdentityBackend) -> identities.IdentityBackend | SomaTorchIdentityBackend:
    identity_backend = identities.prepare_backend(identity_backend, "torch")
    if isinstance(identity_backend, identities.TransferredIdentityBackend):
        return SomaTorchIdentityBackend(identity_backend)
    return identity_backend


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
    out = torch.zeros((batch_size, data.mean_full.shape[0] * 3), dtype=z.dtype, device=z.device)
    index = xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)
    return out.scatter_add(1, index, contrib).reshape(batch_size, data.mean_full.shape[0], 3)
