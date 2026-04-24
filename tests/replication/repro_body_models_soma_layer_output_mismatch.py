#!/usr/bin/env python3
import json

import torch
from body_models.soma.torch import SOMA
from soma import SomaLayer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

body_model = SOMA(model_type="mhr", rotation_type="rotmat").to(device)
soma_layer = SomaLayer(identity_model_type="mhr", device=device)

frames = 2
joints = 77
identity = torch.randn(1, body_model.identity_dim, device=device) * 0.01
scale_params = torch.randn(1, body_model.num_scale_params, device=device) * 0.01
pose = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(frames, joints, 1, 1)
translation = torch.zeros(frames, 3, device=device)

soma_layer.prepare_identity(identity, scale_params=scale_params)

with torch.no_grad():
    layer_vertices = soma_layer.pose(
        pose,
        transl=translation,
        pose2rot=False,
    )["vertices"]

    body_vertices = body_model.forward_vertices(
        pose=pose,
        identity=identity,
        scale_params=scale_params,
        global_translation=translation,
    )

diff = body_vertices - layer_vertices
print(
    json.dumps(
        {
            "body_models_vertices_shape": list(body_vertices.shape),
            "soma_layer_vertices_shape": list(layer_vertices.shape),
            "max_abs_diff": float(diff.abs().max().cpu()),
            "rms_diff": float(diff.square().mean().sqrt().cpu()),
        },
        indent=2,
    )
)
