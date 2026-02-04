"""JAX backend for FLAME model using Flax NNX."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import core
from .io import get_model_path, load_model_data, simplify_mesh, compute_kinematic_fronts

__all__ = ["FLAME"]


class FLAME(nnx.Module):
    """FLAME head model with JAX/Flax NNX backend.

    Args:
        model_path: Path to the FLAME model file or directory.
        simplify: Mesh simplification ratio. 1.0 = original mesh, 2.0 = half faces, etc.
        ground_plane: If True (default), dynamically offset mesh so chin is at Y=0
            regardless of head shape. If False, use native FLAME coordinates.

    Forward API:
        forward_vertices(shape, expression, pose, head_rotation, global_rotation, global_translation)
        forward_skeleton(shape, expression, pose, head_rotation, global_rotation, global_translation)

        shape: [B, 300] shape betas (can use fewer)
        expression: [B, 100] expression coefficients (can use fewer)
        pose: [B, 4, 3] axis-angle for neck, jaw, left_eye, right_eye
        head_rotation: [B, 3] optional root joint rotation (affects skeleton)
        global_rotation: [B, 3] optional post-hoc rotation (doesn't affect skeleton)
        global_translation: [B, 3] optional translation
    """

    NUM_HEAD_JOINTS = 4  # neck, jaw, left_eye, right_eye
    NUM_JOINTS = 5  # root + 4 head joints

    def __init__(
        self,
        model_path: Path | str | None = None,
        simplify: float = 1.0,
        ground_plane: bool = True,
    ):
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)

        # Load full-resolution data
        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)  # (V, 3, 400)
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int32)

        # Fix parent of root (may be -1 or large value in file)
        parents[0] = 0

        # Apply mesh simplification if requested
        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full
            shapedirs = shapedirs_full

        # Store as nnx.Variable for proper pytree handling
        self.v_template = nnx.Variable(jnp.asarray(v_template))
        self.v_template_full = nnx.Variable(jnp.asarray(v_template_full))
        self.lbs_weights = nnx.Variable(jnp.asarray(lbs_weights))
        self.J_regressor = nnx.Variable(jnp.asarray(J_regressor))
        self.parents = nnx.Variable(jnp.asarray(parents))
        self._faces = nnx.Variable(jnp.asarray(faces))

        # FLAME 2023 has combined shape (300) + expression (100) in shapedirs
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full[:, :, :300]))
        self.exprdirs_full = nnx.Variable(jnp.asarray(shapedirs_full[:, :, 300:]))
        self.shapedirs = nnx.Variable(jnp.asarray(shapedirs[:, :, :300]))
        self.exprdirs = nnx.Variable(jnp.asarray(shapedirs[:, :, 300:]))
        self.posedirs = nnx.Variable(jnp.asarray(posedirs.reshape(-1, posedirs.shape[-1]).T))

        self._kinematic_fronts = compute_kinematic_fronts(parents)

    @property
    def faces(self) -> jnp.ndarray:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.v_template[...].shape[0]

    @property
    def skin_weights(self) -> jnp.ndarray:
        return self.lbs_weights[...]

    def forward_vertices(
        self,
        shape: jnp.ndarray,
        expression: jnp.ndarray | None = None,
        pose: jnp.ndarray | None = None,
        head_rotation: jnp.ndarray | None = None,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute mesh vertices [B, V, 3]."""
        B = (
            shape.shape[0]
            if shape.ndim > 1 and shape.shape[0] > 1
            else (pose.shape[0] if pose is not None else 1)
        )

        if expression is None:
            expression = jnp.zeros((B, 100), dtype=jnp.float32)
        if pose is None:
            pose = jnp.zeros((B, self.NUM_HEAD_JOINTS, 3), dtype=jnp.float32)

        return core.forward_vertices(
            v_template=self.v_template[...],
            v_template_full=self.v_template_full[...],
            shapedirs=self.shapedirs[...],
            shapedirs_full=self.shapedirs_full[...],
            exprdirs=self.exprdirs[...],
            exprdirs_full=self.exprdirs_full[...],
            posedirs=self.posedirs[...],
            lbs_weights=self.lbs_weights[...],
            J_regressor=self.J_regressor[...],
            parents=self.parents[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def forward_skeleton(
        self,
        shape: jnp.ndarray,
        expression: jnp.ndarray | None = None,
        pose: jnp.ndarray | None = None,
        head_rotation: jnp.ndarray | None = None,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute skeleton joint transforms [B, 5, 4, 4]."""
        B = (
            shape.shape[0]
            if shape.ndim > 1 and shape.shape[0] > 1
            else (pose.shape[0] if pose is not None else 1)
        )

        if expression is None:
            expression = jnp.zeros((B, 100), dtype=jnp.float32)
        if pose is None:
            pose = jnp.zeros((B, self.NUM_HEAD_JOINTS, 3), dtype=jnp.float32)

        return core.forward_skeleton(
            v_template_full=self.v_template_full[...],
            shapedirs_full=self.shapedirs_full[...],
            exprdirs_full=self.exprdirs_full[...],
            J_regressor=self.J_regressor[...],
            parents=self.parents[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            expression=expression,
            pose=pose,
            head_rotation=head_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        """Get rest pose parameters."""
        return {
            "shape": jnp.zeros((1, 300), dtype=dtype),
            "expression": jnp.zeros((batch_size, 100), dtype=dtype),
            "pose": jnp.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), dtype=dtype),
            "head_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
