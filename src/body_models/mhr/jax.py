"""JAX backend for MHR model using Flax NNX with neural pose correctives."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from . import core
from .io import get_model_path, load_model_data, load_pose_correctives_weights, compute_kinematic_fronts, simplify_mesh

__all__ = ["MHR"]


class MHR(BodyModel, nnx.Module):
    """MHR body model with JAX/Flax NNX backend and neural pose correctives.

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

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        lod: int = 1,
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

            base_vertices = new_vertices.astype(np.float32)
            blendshape_dirs = blendshape_dirs_full[:, vertex_map]
            skin_weights = skin_weights_full[vertex_map]
            skin_indices = skin_indices_full[vertex_map]
            faces = new_faces.astype(np.int64)
        else:
            base_vertices = base_vertices_full
            blendshape_dirs = blendshape_dirs_full
            skin_weights = skin_weights_full
            skin_indices = skin_indices_full
            faces = faces.astype(np.int64)

        # Store as nnx.Variable for proper pytree handling
        self.base_vertices = nnx.Variable(jnp.asarray(base_vertices))
        self.blendshape_dirs = nnx.Variable(jnp.asarray(blendshape_dirs))
        self._skin_weights = nnx.Variable(jnp.asarray(skin_weights))
        self._skin_indices = nnx.Variable(jnp.asarray(skin_indices))
        self._faces = nnx.Variable(jnp.asarray(faces))

        # Skeleton data
        self.joint_offsets = nnx.Variable(jnp.asarray(data["joint_offsets"].numpy()))
        self.joint_pre_rotations = nnx.Variable(jnp.asarray(data["joint_pre_rotations"].numpy()))
        self.parameter_transform = nnx.Variable(jnp.asarray(data["parameter_transform"].numpy()))

        inv_bind = data["inverse_bind_pose"].numpy()
        t, q, s = inv_bind[..., :3], inv_bind[..., 3:7], inv_bind[..., 7:8]
        self.bind_inv_linear = nnx.Variable(jnp.asarray(SO3.to_matrix(q, xyzw=True) * s[..., None]))
        self.bind_inv_translation = nnx.Variable(jnp.asarray(t))

        # Load pose correctives weights
        corrective_weights = load_pose_correctives_weights(resolved_path, lod)
        self.corrective_W1 = nnx.Variable(jnp.asarray(corrective_weights["W1"]))
        self.corrective_W2 = nnx.Variable(jnp.asarray(corrective_weights["W2"]))

        joint_parents = data["joint_parents"]
        self._kinematic_fronts = compute_kinematic_fronts(joint_parents)
        self._num_joints = len(joint_parents)
        self._pose_dim = data["parameter_transform"].shape[1] - self.SHAPE_DIM

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def num_vertices(self) -> int:
        return self.base_vertices[...].shape[0]

    @property
    def pose_dim(self) -> int:
        return self._pose_dim

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.base_vertices[...] * 0.01

    @property
    def skin_weights(self) -> Float[jax.Array, "V K"]:
        return self._skin_weights[...]

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 45"],
        pose: Float[jax.Array, "B 204"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        """Compute mesh vertices [B, V, 3] in meters."""
        return core.forward_vertices(
            base_vertices=self.base_vertices[...],
            blendshape_dirs=self.blendshape_dirs[...],
            skin_weights=self._skin_weights[...],
            skin_indices=self._skin_indices[...],
            joint_offsets=self.joint_offsets[...],
            joint_pre_rotations=self.joint_pre_rotations[...],
            parameter_transform=self.parameter_transform[...],
            bind_inv_linear=self.bind_inv_linear[...],
            bind_inv_translation=self.bind_inv_translation[...],
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            expr_dim=self.EXPR_DIM,
            shape=shape,
            pose=pose,
            expression=expression,
            global_rotation=global_rotation,
            global_translation=global_translation,
            corrective_W1=self.corrective_W1[...],
            corrective_W2=self.corrective_W2[...],
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 45"],
        pose: Float[jax.Array, "B 204"],
        expression: Float[jax.Array, "B 72"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B J 4 4"]:
        """Compute skeleton transforms [B, J, 4, 4] in meters."""
        return core.forward_skeleton(
            joint_offsets=self.joint_offsets[...],
            joint_pre_rotations=self.joint_pre_rotations[...],
            parameter_transform=self.parameter_transform[...],
            kinematic_fronts=self._kinematic_fronts,
            num_joints=self.num_joints,
            shape_dim=self.SHAPE_DIM,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": jnp.zeros((batch_size, self.pose_dim), dtype=dtype),
            "expression": jnp.zeros((batch_size, self.EXPR_DIM), dtype=dtype),
            "global_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
