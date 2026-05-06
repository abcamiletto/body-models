"""PyTorch SOMA backend."""

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from ..io import SomaWeights
from . import core

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "PreparedSomaIdentity",
    "prepare_data",
]

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
linear_blend_skinning = core.linear_blend_skinning
PreparedSomaIdentity = core.PreparedSomaIdentity


class SomaTorchCorrectives(nn.Module):
    corrective_bindpose: Float[torch.Tensor, "Jf 3 3"]
    corrective_W1: Float[torch.Tensor, "D K"]
    corrective_W2_rows: Int[torch.Tensor, "NNZ"]
    corrective_W2_cols: Int[torch.Tensor, "NNZ"]
    corrective_W2_values: Float[torch.Tensor, "NNZ"]

    def __init__(self, correctives):
        super().__init__()
        self.corrective_bindpose = nn.Buffer(torch.as_tensor(correctives.corrective_bindpose))
        self.corrective_W1 = nn.Buffer(torch.as_tensor(correctives.corrective_W1))
        corrective_W2_rows = torch.as_tensor(correctives.corrective_W2_rows, dtype=torch.int64)
        corrective_W2_cols = torch.as_tensor(correctives.corrective_W2_cols, dtype=torch.int64)
        self.corrective_W2_rows = nn.Buffer(corrective_W2_rows)
        self.corrective_W2_cols = nn.Buffer(corrective_W2_cols)
        self.corrective_W2_values = nn.Buffer(torch.as_tensor(correctives.corrective_W2_values))


class SomaTorchTopology(nn.Module):
    joint_children_indices_full: Int[torch.Tensor, "Jf C"]
    skinned_vertex_indices_full_index: Int[torch.Tensor, "Jf K"]

    def __init__(self, topology):
        super().__init__()
        self.parents_full = topology.parents_full
        self.joint_children_full = topology.joint_children_full
        self.skinned_vertex_indices_full = topology.skinned_vertex_indices_full
        self.kinematic_fronts_full = topology.kinematic_fronts_full
        self.joint_children_indices_full = nn.Buffer(torch.as_tensor(topology.joint_children_indices_full))
        self.skinned_vertex_indices_full_index = nn.Buffer(
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

        self.mean_full = nn.Buffer(torch.as_tensor(weights.mean_full))
        self.mean_active = nn.Buffer(torch.as_tensor(weights.mean_active))
        self.shapedirs_full = nn.Buffer(torch.as_tensor(weights.shapedirs_full))
        self.shapedirs_active = nn.Buffer(torch.as_tensor(weights.shapedirs_active))
        self.eigenvalues = nn.Buffer(torch.as_tensor(weights.eigenvalues))
        self.bind_shape_full = nn.Buffer(torch.as_tensor(weights.bind_shape_full))
        self.bind_pose_world = nn.Buffer(torch.as_tensor(weights.bind_pose_world))
        self.bind_pose_local = nn.Buffer(torch.as_tensor(weights.bind_pose_local))
        self.t_pose_world = nn.Buffer(torch.as_tensor(weights.t_pose_world))
        self.t_pose_local = nn.Buffer(torch.as_tensor(weights.t_pose_local))
        self.joint_regressor = nn.Buffer(torch.as_tensor(weights.joint_regressor))
        self.skin_weights_full = nn.Buffer(torch.as_tensor(weights.skin_weights_full))
        self.skin_weights_active = nn.Buffer(torch.as_tensor(weights.skin_weights_active))
        self.faces = nn.Buffer(torch.as_tensor(weights.faces, dtype=torch.int64))
        facial_inner_vertices = torch.as_tensor(weights.facial_inner_vertices, dtype=torch.int64)
        vertex_map = None
        if weights.vertex_map is not None:
            vertex_map = torch.as_tensor(weights.vertex_map, dtype=torch.int64)
        self.facial_inner_vertices = nn.Buffer(facial_inner_vertices)
        self.vertex_map = nn.Buffer(vertex_map) if vertex_map is not None else None


def prepare_data(weights: SomaWeights) -> SomaTorchWeights:
    return SomaTorchWeights(weights)


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
    out_shape = batch_size, data.mean_full.shape[0] * 3
    out = torch.zeros(out_shape, dtype=z.dtype, device=z.device)
    index = xp.broadcast_to(correctives.corrective_W2_cols[None], contrib.shape)
    return out.scatter_add(1, index, contrib).reshape(batch_size, data.mean_full.shape[0], 3)
