# Derived from: https://github.com/facebookresearch/MHR
# Original license: Apache 2.0 (https://github.com/facebookresearch/MHR/blob/main/LICENSE)

"""PyTorch backend for MHR model with neural pose correctives."""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from nanomanifold import SO3
from torch import Tensor

from ..base import BodyModel
from . import core
from .io import get_model_path, load_model_data, load_pose_correctives, compute_kinematic_fronts, simplify_mesh


class MHR(BodyModel, nn.Module):
    """MHR body model with PyTorch backend and neural pose correctives.

    Args:
        model_path: Path to MHR model directory. Auto-downloads if None.
        lod: Level of detail for pose correctives (1 = default).
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.

    Forward API:
        forward_vertices(shape, pose, expression, global_rotation, global_translation)
        forward_skeleton(shape, pose, expression, global_rotation, global_translation)

        shape: [B, 45] identity blendshapes
        pose: [B, 204] joint parameters
        expression: [B, 72] facial blendshapes (optional)
    """

    SHAPE_DIM = 45
    EXPR_DIM = 72
    _PARAMS_PER_JOINT = 7

    # Type declarations for registered buffers
    base_vertices: Tensor
    base_vertices_full: Tensor
    blendshape_dirs: Tensor
    blendshape_dirs_full: Tensor
    _skin_weights: Tensor
    _skin_weights_full: Tensor
    _skin_indices: Tensor
    _skin_indices_full: Tensor
    _vertex_map: Tensor | None
    joint_offsets: Tensor
    joint_pre_rotations: Tensor
    parameter_transform: Tensor
    bind_inv_linear: Tensor
    bind_inv_translation: Tensor
    _faces: Tensor

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
        simplify: float = 1.0,
    ) -> None:
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        super().__init__()

        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        base_vertices_full = data["base_vertices"]
        blendshape_dirs_full = data["blendshape_dirs"]
        skin_weights_full = data["skin_weights"]
        skin_indices_full = data["skin_indices"].to(torch.int64)
        faces = data["faces"]

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            vertices_np = base_vertices_full.numpy()
            faces_np = faces.numpy().astype(int)
            new_vertices, new_faces, vertex_map = simplify_mesh(vertices_np, faces_np, target_faces)

            self.register_buffer("base_vertices", torch.as_tensor(new_vertices, dtype=base_vertices_full.dtype))
            self.register_buffer("blendshape_dirs", blendshape_dirs_full[:, vertex_map].clone())
            self.register_buffer("_skin_weights", skin_weights_full[vertex_map].clone())
            self.register_buffer("_skin_indices", skin_indices_full[vertex_map].clone())
            self.register_buffer("_faces", torch.as_tensor(new_faces, dtype=torch.int64))
            self.register_buffer("_vertex_map", torch.as_tensor(vertex_map, dtype=torch.int64))
        else:
            self.register_buffer("base_vertices", base_vertices_full)
            self.register_buffer("blendshape_dirs", blendshape_dirs_full)
            self.register_buffer("_skin_weights", skin_weights_full)
            self.register_buffer("_skin_indices", skin_indices_full)
            self.register_buffer("_faces", faces)
            self._vertex_map = None

        # Full-resolution buffers for pose correctives (always needed)
        self.register_buffer("base_vertices_full", base_vertices_full)
        self.register_buffer("blendshape_dirs_full", blendshape_dirs_full)
        self.register_buffer("_skin_weights_full", skin_weights_full)
        self.register_buffer("_skin_indices_full", skin_indices_full)

        # Skeleton buffers
        self.register_buffer("joint_offsets", data["joint_offsets"])
        self.register_buffer("joint_pre_rotations", data["joint_pre_rotations"])
        self.register_buffer("parameter_transform", data["parameter_transform"])

        inv_bind = data["inverse_bind_pose"]
        t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
        self.register_buffer("bind_inv_linear", SO3.to_matrix(q, xyzw=True) * s.unsqueeze(-1))
        self.register_buffer("bind_inv_translation", t)

        self._pose_correctives = load_pose_correctives(resolved_path, lod)
        self._kinematic_fronts = compute_kinematic_fronts(data["joint_parents"])

    @property
    def faces(self) -> Int[Tensor, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.joint_offsets.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.base_vertices.shape[0]

    @property
    def pose_dim(self) -> int:
        return self.parameter_transform.shape[1] - self.SHAPE_DIM

    @property
    def skin_weights(self) -> Float[Tensor, "V J"]:
        V, K = self._skin_weights.shape
        dense = torch.zeros(V, self.num_joints, device=self._skin_weights.device, dtype=self._skin_weights.dtype)
        dense.scatter_(1, self._skin_indices, self._skin_weights)
        return dense

    @property
    def rest_vertices(self) -> Float[Tensor, "V 3"]:
        return self.base_vertices * 0.01

    def forward_vertices(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute mesh vertices [B, V, 3] in meters."""
        # For simplified meshes, compute correctives at full resolution then downsample
        if self._vertex_map is not None:
            return self._forward_vertices_simplified(shape, pose, expression, global_rotation, global_translation)

        return core.forward_vertices(
            base_vertices=self.base_vertices,
            blendshape_dirs=self.blendshape_dirs,
            skin_weights=self._skin_weights,
            skin_indices=self._skin_indices,
            joint_offsets=self.joint_offsets,
            joint_pre_rotations=self.joint_pre_rotations,
            parameter_transform=self.parameter_transform,
            bind_inv_linear=self.bind_inv_linear,
            bind_inv_translation=self.bind_inv_translation,
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            expr_dim=self.EXPR_DIM,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            pose_correctives_fn=self._pose_correctives,
        )

    def _forward_vertices_simplified(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B V 3"]:
        """Compute vertices for simplified mesh (correctives computed at full resolution)."""
        B = pose.shape[0]
        dtype = pose.dtype

        if expression is None:
            expression = torch.zeros((B, self.EXPR_DIM), device=pose.device, dtype=dtype)

        # Broadcast shape/expression
        if shape.shape[0] == 1 and B > 1:
            shape = shape.expand(B, -1)
        if expression.shape[0] == 1 and B > 1:
            expression = expression.expand(B, -1)

        coeffs = torch.cat([shape, expression], dim=1)

        # Forward skeleton to get joint params
        _, _, _, j_p = core._forward_skeleton_core(
            xp=torch,
            pose=pose,
            joint_offsets=self.joint_offsets,
            joint_pre_rotations=self.joint_pre_rotations,
            parameter_transform=self.parameter_transform,
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
        )

        # Compute full-resolution v_t with correctives
        v_t_full = self.base_vertices_full + torch.einsum("bi,ivk->bvk", coeffs, self.blendshape_dirs_full)
        v_t_full = v_t_full + self._pose_correctives(j_p)

        # Downsample to simplified vertices
        v_t = v_t_full[:, self._vertex_map]

        # Now do skinning with simplified buffers
        return self._apply_skinning(v_t, pose, global_rotation, global_translation)

    def _apply_skinning(
        self,
        v_t: Float[Tensor, "B V 3"],
        pose: Float[Tensor, "B 204"],
        global_rotation: Float[Tensor, "B 3"] | None,
        global_translation: Float[Tensor, "B 3"] | None,
    ) -> Float[Tensor, "B V 3"]:
        """Apply linear blend skinning to shaped vertices."""
        t_g, r_g, s_g, _ = core._forward_skeleton_core(
            xp=torch,
            pose=pose,
            joint_offsets=self.joint_offsets,
            joint_pre_rotations=self.joint_pre_rotations,
            parameter_transform=self.parameter_transform,
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
        )

        # Linear blend skinning
        lin_g = r_g * s_g.unsqueeze(-1)
        lin = torch.einsum("bjik,jkl->bjil", lin_g, self.bind_inv_linear)
        t = torch.einsum("bjik,jk->bji", lin_g, self.bind_inv_translation) + t_g

        lin = lin[:, self._skin_indices]
        t = t[:, self._skin_indices]
        v_transformed = torch.einsum("bvkij,bvj->bvki", lin, v_t) + t
        verts = (v_transformed * self._skin_weights[None, :, :, None]).sum(dim=2)

        # Convert to meters
        verts = verts * 0.01

        # Apply global transform
        if global_rotation is not None:
            R = SO3.to_matrix(SO3.from_axis_angle(global_rotation, xp=torch), xp=torch)
            verts = verts @ R.mT
        if global_translation is not None:
            verts = verts + global_translation[:, None]

        return verts

    def forward_skeleton(
        self,
        shape: Float[Tensor, "B|1 45"],
        pose: Float[Tensor, "B 204"],
        expression: Float[Tensor, "B 72"] | None = None,
        global_rotation: Float[Tensor, "B 3"] | None = None,
        global_translation: Float[Tensor, "B 3"] | None = None,
    ) -> Float[Tensor, "B J 4 4"]:
        """Compute skeleton transforms [B, J, 4, 4] in meters."""
        return core.forward_skeleton(
            joint_offsets=self.joint_offsets,
            joint_pre_rotations=self.joint_pre_rotations,
            parameter_transform=self.parameter_transform,
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        device = self.base_vertices.device
        return {
            "shape": torch.zeros((1, self.SHAPE_DIM), device=device, dtype=dtype),
            "pose": torch.zeros((batch_size, self.pose_dim), device=device, dtype=dtype),
            "expression": torch.zeros((batch_size, self.EXPR_DIM), device=device, dtype=dtype),
            "global_rotation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
            "global_translation": torch.zeros((batch_size, 3), device=device, dtype=dtype),
        }
