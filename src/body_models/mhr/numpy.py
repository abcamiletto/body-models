"""NumPy backend for MHR model.

Note: Pose correctives are NOT available in this backend.
Use the PyTorch backend for full accuracy with pose correctives.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from . import core
from .io import get_model_path, load_model_data, compute_kinematic_fronts, simplify_mesh

__all__ = ["MHR"]


class MHR(BodyModel):
    """MHR body model with NumPy backend (no pose correctives).

    Args:
        model_path: Path to MHR model directory. Auto-downloads if None.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.

    Note:
        This backend does NOT include neural pose correctives.
        For full accuracy, use the PyTorch backend.

    Forward API:
        forward_vertices(shape, pose, expression, global_rotation, global_translation)
        forward_skeleton(shape, pose, expression, global_rotation, global_translation)

        shape: [B, 45] identity blendshapes
        pose: [B, 204] joint parameters
        expression: [B, 72] facial blendshapes (optional)
    """

    SHAPE_DIM = 45
    EXPR_DIM = 72

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        simplify: float = 1.0,
    ) -> None:
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"

        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        base_vertices_full = data["base_vertices"].numpy()
        blendshape_dirs_full = data["blendshape_dirs"].numpy()
        skin_weights_full = data["skin_weights"].numpy()
        skin_indices_full = data["skin_indices"].numpy().astype(np.int64)
        faces = data["faces"].numpy()

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            new_vertices, new_faces, vertex_map = simplify_mesh(base_vertices_full, faces.astype(int), target_faces)

            self.base_vertices = new_vertices.astype(np.float32)
            self.blendshape_dirs = blendshape_dirs_full[:, vertex_map]
            self._skin_weights = skin_weights_full[vertex_map]
            self._skin_indices = skin_indices_full[vertex_map]
            self._faces = new_faces.astype(np.int64)
        else:
            self.base_vertices = base_vertices_full
            self.blendshape_dirs = blendshape_dirs_full
            self._skin_weights = skin_weights_full
            self._skin_indices = skin_indices_full
            self._faces = faces.astype(np.int64)

        # Skeleton data
        self.joint_offsets = data["joint_offsets"].numpy()
        self.joint_pre_rotations = data["joint_pre_rotations"].numpy()
        self.parameter_transform = data["parameter_transform"].numpy()

        inv_bind = data["inverse_bind_pose"].numpy()
        t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
        self.bind_inv_linear = SO3.to_matrix(q, xyzw=True) * s[..., None]
        self.bind_inv_translation = t

        self._kinematic_fronts = compute_kinematic_fronts(data["joint_parents"])

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
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
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        return self.base_vertices * 0.01

    @property
    def skin_weights(self) -> Float[np.ndarray, "V K"]:
        return self._skin_weights

    def forward_vertices(
        self,
        shape: Float[np.ndarray, "B|1 45"],
        pose: Float[np.ndarray, "B 204"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        """Compute mesh vertices [B, V, 3] in meters.

        Note: Pose correctives are NOT applied in this backend.
        """
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
            pose_correctives_fn=None,  # No pose correctives for NumPy backend
        )

    def forward_skeleton(
        self,
        shape: Float[np.ndarray, "B|1 45"],
        pose: Float[np.ndarray, "B 204"],
        expression: Float[np.ndarray, "B 72"] | None = None,
        global_rotation: Float[np.ndarray, "B 3"] | None = None,
        global_translation: Float[np.ndarray, "B 3"] | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
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

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, np.ndarray]:
        return {
            "shape": np.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": np.zeros((batch_size, self.pose_dim), dtype=dtype),
            "expression": np.zeros((batch_size, self.EXPR_DIM), dtype=dtype),
            "global_rotation": np.zeros((batch_size, 3), dtype=dtype),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
