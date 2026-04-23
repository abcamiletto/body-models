"""JAX backend for SOMA model using Flax NNX."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int
from nanomanifold import SO3

from ..base import BodyModel
from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import compute_kinematic_fronts, get_model_path, load_model_data, load_pose_correctives_weights, simplify_mesh

__all__ = ["SOMA"]


class SOMA(BodyModel, nnx.Module):
    """SOMA body model with JAX/Flax NNX backend."""

    SHAPE_DIM = 128
    NUM_JOINTS = 77

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        simplify: float = 1.0,
        rotation_type: core.RotationType = "axis_angle",
    ) -> None:
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0, "simplify must be >= 1.0 (1.0 = original mesh)"

        self.rotation_type = rotation_type
        resolved_path = get_model_path(model_path)
        data = load_model_data(resolved_path)
        corrective_weights = load_pose_correctives_weights(resolved_path)

        mean_full = data["mean"]
        shapedirs_full = data["shapedirs"]
        faces = data["faces"]
        skin_weights_full = data["skin_weights_full"]

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            mean_active, faces, vertex_map = simplify_mesh(mean_full, faces.astype(int), target_faces)
            shapedirs_active = shapedirs_full[:, vertex_map]
            skin_weights_active = skin_weights_full[vertex_map]
            self._vertex_map = nnx.Variable(jnp.asarray(np.asarray(vertex_map, dtype=np.int64)))
        else:
            mean_active = mean_full
            shapedirs_active = shapedirs_full
            skin_weights_active = skin_weights_full
            self._vertex_map = None

        self.mean_full = nnx.Variable(jnp.asarray(mean_full))
        self.mean_active = nnx.Variable(jnp.asarray(mean_active))
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full))
        self.shapedirs_active = nnx.Variable(jnp.asarray(shapedirs_active))
        self.eigenvalues = nnx.Variable(jnp.asarray(data["eigenvalues"]))
        self.bind_shape_full = nnx.Variable(jnp.asarray(data["bind_shape"]))
        self.bind_pose_world = nnx.Variable(jnp.asarray(data["bind_pose_world"]))
        self.bind_pose_local = nnx.Variable(jnp.asarray(data["bind_pose_local"]))
        self.t_pose_world = nnx.Variable(jnp.asarray(data["t_pose_world"]))
        self.joint_regressor = nnx.Variable(jnp.asarray(data["joint_regressor"]))
        self.corrective_bindpose = nnx.Variable(jnp.asarray(corrective_weights["bindpose"]))
        self.corrective_W1 = nnx.Variable(jnp.asarray(corrective_weights["W1"]))
        self.corrective_W2_rows = nnx.Variable(jnp.asarray(corrective_weights["W2_rows"]))
        self.corrective_W2_cols = nnx.Variable(jnp.asarray(corrective_weights["W2_cols"]))
        self.corrective_W2_values = nnx.Variable(jnp.asarray(corrective_weights["W2_values"]))
        self._corrective_use_tanh = bool(corrective_weights["use_tanh"])
        self._skin_weights_full = nnx.Variable(jnp.asarray(skin_weights_full))
        self._skin_weights_active = nnx.Variable(jnp.asarray(skin_weights_active))
        self._faces = nnx.Variable(jnp.asarray(np.asarray(faces, dtype=np.int64)))

        self.parents = list(data["parents"])
        self._parents_full = data["joint_parents_full"].tolist()
        self._joint_children_full = data["joint_children_full"]
        self._skinned_vertex_indices_full = data["skinned_vertex_indices_full"]
        self._kinematic_fronts_full = compute_kinematic_fronts(self._parents_full)
        self._joint_names = list(data["joint_names"])

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.mean_active[...].shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        return self._skin_weights_active[...][:, 1:]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.mean_active[...] * 0.01

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 128"],
        pose: Float[jax.Array, "B 77 N"] | Float[jax.Array, "B 77 3 3"],
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        vertex_indices=None,
        apply_correctives: bool = True,
    ) -> Float[jax.Array, "B V 3"]:
        return core.forward_vertices(
            mean_full=self.mean_full[...],
            mean_active=self.mean_active[...],
            shapedirs_full=self.shapedirs_full[...],
            shapedirs_active=self.shapedirs_active[...],
            eigenvalues=self.eigenvalues[...],
            bind_shape_full=self.bind_shape_full[...],
            skin_weights_active=self._skin_weights_active[...],
            bind_pose_world=self.bind_pose_world[...],
            bind_pose_local=self.bind_pose_local[...],
            t_pose_world=self.t_pose_world[...],
            joint_regressor=self.joint_regressor[...],
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            vertex_map=None if self._vertex_map is None else self._vertex_map[...],
            corrective_bindpose=self.corrective_bindpose[...],
            corrective_W1=self.corrective_W1[...],
            corrective_W2_rows=self.corrective_W2_rows[...],
            corrective_W2_cols=self.corrective_W2_cols[...],
            corrective_W2_values=self.corrective_W2_values[...],
            corrective_use_tanh=self._corrective_use_tanh,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 128"],
        pose: Float[jax.Array, "B 77 N"] | Float[jax.Array, "B 77 3 3"],
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
        joint_indices=None,
        apply_correctives: bool = True,
    ) -> Float[jax.Array, "B 77 4 4"]:
        return core.forward_skeleton(
            mean_full=self.mean_full[...],
            shapedirs_full=self.shapedirs_full[...],
            eigenvalues=self.eigenvalues[...],
            bind_shape_full=self.bind_shape_full[...],
            skin_weights_full=self._skin_weights_full[...],
            bind_pose_world=self.bind_pose_world[...],
            bind_pose_local=self.bind_pose_local[...],
            t_pose_world=self.t_pose_world[...],
            joint_regressor=self.joint_regressor[...],
            joint_children_full=self._joint_children_full,
            skinned_vertex_indices_full=self._skinned_vertex_indices_full,
            kinematic_fronts_full=self._kinematic_fronts_full,
            parents_full=self._parents_full,
            shape=shape,
            pose=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            apply_correctives=apply_correctives,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((batch_size, self.num_joints, 3), dtype=dtype)
        rot_ref = jnp.zeros((batch_size, 3), dtype=dtype)
        return {
            "shape": jnp.zeros((1, self.SHAPE_DIM), dtype=dtype),
            "pose": SO3.identity_as(
                pose_ref,
                batch_dims=(batch_size, self.num_joints),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                rot_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
