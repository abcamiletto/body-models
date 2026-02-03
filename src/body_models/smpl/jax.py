"""JAX backend for SMPL model using Flax NNX."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from . import core
from .io import compute_kinematic_fronts, get_model_path, load_model_data, simplify_mesh

__all__ = ["SMPL"]


class SMPL(nnx.Module):
    """SMPL body model with JAX/Flax NNX backend."""

    NUM_BODY_JOINTS = 23
    NUM_JOINTS = 24

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: str | None = None,
        simplify: float = 1.0,
        ground_plane: bool = True,
    ):
        if gender is not None and gender not in ("neutral", "male", "female"):
            raise ValueError(f"Invalid gender: {gender}. Must be 'neutral', 'male', or 'female'.")
        assert simplify >= 1.0

        # Default gender to "neutral" for attribute storage when model_path is given
        self.gender = gender if gender is not None else "neutral"
        self.ground_plane = ground_plane

        resolved_path = get_model_path(model_path, gender)
        data = load_model_data(resolved_path)

        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
        shapedirs = shapedirs_full
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int32)

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full

        # Store as nnx.Variable for proper pytree handling
        self.v_template = nnx.Variable(jnp.asarray(v_template))
        self.v_template_full = nnx.Variable(jnp.asarray(v_template_full))
        self.shapedirs = nnx.Variable(jnp.asarray(shapedirs))
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full))
        self.posedirs = nnx.Variable(jnp.asarray(posedirs.reshape(-1, posedirs.shape[-1]).T))
        self.lbs_weights = nnx.Variable(jnp.asarray(lbs_weights))
        self.J_regressor = nnx.Variable(jnp.asarray(J_regressor))
        self.parents = nnx.Variable(jnp.asarray(parents))
        self._faces = nnx.Variable(jnp.asarray(faces))
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

    def forward_vertices(
        self,
        shape: jnp.ndarray,
        body_pose: jnp.ndarray,
        pelvis_rotation: jnp.ndarray | None = None,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return core.forward_vertices(
            v_template=self.v_template[...],
            v_template_full=self.v_template_full[...],
            shapedirs=self.shapedirs[...],
            shapedirs_full=self.shapedirs_full[...],
            posedirs=self.posedirs[...],
            lbs_weights=self.lbs_weights[...],
            J_regressor=self.J_regressor[...],
            parents=self.parents[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def forward_skeleton(
        self,
        shape: jnp.ndarray,
        body_pose: jnp.ndarray,
        pelvis_rotation: jnp.ndarray | None = None,
        global_rotation: jnp.ndarray | None = None,
        global_translation: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return core.forward_skeleton(
            v_template_full=self.v_template_full[...],
            shapedirs_full=self.shapedirs_full[...],
            J_regressor=self.J_regressor[...],
            parents=self.parents[...],
            kinematic_fronts=self._kinematic_fronts,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jnp.ndarray]:
        return {
            "shape": jnp.zeros((1, 10), dtype=dtype),
            "body_pose": jnp.zeros((batch_size, self.NUM_BODY_JOINTS, 3), dtype=dtype),
            "pelvis_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
