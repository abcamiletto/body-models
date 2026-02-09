"""JAX backend for SMPL-X model using Flax NNX."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Float, Int

from ..base import BodyModel
from . import core
from .io import compute_kinematic_fronts, get_model_path, load_model_data, simplify_mesh

__all__ = ["SMPLX"]


class SMPLX(BodyModel, nnx.Module):
    """SMPL-X body model with JAX/Flax NNX backend."""

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30  # 15 per hand
    NUM_HEAD_JOINTS = 3  # jaw, left eye, right eye
    NUM_JOINTS = 55  # 22 body + 30 hands + 3 head

    def __init__(
        self,
        model_path: Path | str | None = None,
        gender: str | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        ground_plane: bool = True,
        use_hand_pca: bool = False,  # Accepted for compatibility, not used
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
        self.lbs_weights = nnx.Variable(jnp.asarray(lbs_weights))
        self.J_regressor = nnx.Variable(jnp.asarray(J_regressor))
        self.parents = nnx.Variable(jnp.asarray(parents))
        self._faces = nnx.Variable(jnp.asarray(faces))

        # Hand pose mean
        hand_mean = np.stack(
            [
                np.asarray(data["hands_meanl"], dtype=np.float32),
                np.asarray(data["hands_meanr"], dtype=np.float32),
            ]
        )
        if flat_hand_mean:
            hand_mean = np.zeros_like(hand_mean)
        self.hand_mean = nnx.Variable(jnp.asarray(hand_mean))

        # Blend shapes - split into shape and expression
        self.shapedirs = nnx.Variable(jnp.asarray(shapedirs[:, :, :300]))
        self.shapedirs_full = nnx.Variable(jnp.asarray(shapedirs_full[:, :, :300]))
        self.exprdirs = nnx.Variable(jnp.asarray(shapedirs[:, :, 300:400]))
        self.exprdirs_full = nnx.Variable(jnp.asarray(shapedirs_full[:, :, 300:400]))
        self.posedirs = nnx.Variable(jnp.asarray(posedirs.reshape(-1, posedirs.shape[-1]).T))

        self._kinematic_fronts = compute_kinematic_fronts(parents)

        # Precompute Y offset for ground plane (min Y of rest pose mesh)
        self._rest_pose_y_offset = float(-v_template_full[:, 1].min())

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self._faces[...]

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def num_vertices(self) -> int:
        return self.v_template[...].shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V 55"]:
        return self.lbs_weights[...]

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        return self.v_template[...]

    def forward_vertices(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 21 3"],
        hand_pose: Float[jax.Array, "B 30 3"],
        head_pose: Float[jax.Array, "B 3 3"],
        expression: Float[jax.Array, "B 10"] | None = None,
        pelvis_rotation: Float[jax.Array, "B 3"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
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
            hand_mean=self.hand_mean[...],
            rest_pose_y_offset=self._rest_pose_y_offset,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def forward_skeleton(
        self,
        shape: Float[jax.Array, "B|1 10"],
        body_pose: Float[jax.Array, "B 21 3"],
        hand_pose: Float[jax.Array, "B 30 3"],
        head_pose: Float[jax.Array, "B 3 3"],
        expression: Float[jax.Array, "B 10"] | None = None,
        pelvis_rotation: Float[jax.Array, "B 3"] | None = None,
        global_rotation: Float[jax.Array, "B 3"] | None = None,
        global_translation: Float[jax.Array, "B 3"] | None = None,
    ) -> Float[jax.Array, "B 55 4 4"]:
        return core.forward_skeleton(
            v_template_full=self.v_template_full[...],
            shapedirs_full=self.shapedirs_full[...],
            exprdirs_full=self.exprdirs_full[...],
            J_regressor=self.J_regressor[...],
            parents=self.parents[...],
            kinematic_fronts=self._kinematic_fronts,
            hand_mean=self.hand_mean[...],
            rest_pose_y_offset=self._rest_pose_y_offset,
            shape=shape,
            body_pose=body_pose,
            hand_pose=hand_pose,
            head_pose=head_pose,
            expression=expression,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            ground_plane=self.ground_plane,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=jnp.float32) -> dict[str, jax.Array]:
        return {
            "shape": jnp.zeros((1, 10), dtype=dtype),
            "body_pose": jnp.zeros((batch_size, self.NUM_BODY_JOINTS, 3), dtype=dtype),
            "hand_pose": jnp.zeros((batch_size, self.NUM_HAND_JOINTS, 3), dtype=dtype),
            "head_pose": jnp.zeros((batch_size, self.NUM_HEAD_JOINTS, 3), dtype=dtype),
            "expression": jnp.zeros((batch_size, 10), dtype=dtype),
            "pelvis_rotation": jnp.zeros((batch_size, 3), dtype=dtype),
            "global_translation": jnp.zeros((batch_size, 3), dtype=dtype),
        }
