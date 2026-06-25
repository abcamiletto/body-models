"""JAX backend for the procedural SMPL humanoid robot."""

from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models import common
from body_models.base import BodyModel
from body_models.robots.smpl_humanoid.backends import core
from body_models.robots.smpl_humanoid.backends import jax as backend
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import ACTION_SIZE, QPOS_SIZE, load_model_data

__all__ = ["SmplHumanoid"]
BODY_POSE_ORDER = [smpl_index for _, smpl_index in BODY_JOINTS]


class SmplHumanoid(BodyModel):
    """Rigid procedural humanoid using the canonical 24-joint SMPL hierarchy."""

    is_rigid_body = True
    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self, model_path: Path | str | None = None, *, rotation_type: core.RotationType = "axis_angle"
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES or rotation_type == "hinge":
            raise ValueError(f"Invalid rotation_type for SmplHumanoid: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.weights = common.jaxify(load_model_data(model_path))
        self.pd_action_offset = jnp.zeros(ACTION_SIZE, dtype=jnp.float32)
        self.pd_action_scale = jnp.full((ACTION_SIZE,), jnp.pi, dtype=jnp.float32)

    @property
    def faces(self) -> Int[jax.Array, "F 3"]:
        return self.weights.faces

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def parents(self) -> list[int]:
        return self.weights.parents

    @property
    def qpos_joint_names(self) -> list[str]:
        return self.weights.qpos_joint_names

    @property
    def qpos_joint_indices(self) -> list[int]:
        return self.weights.qpos_joint_indices

    @property
    def qpos_joint_limits(self) -> Float[jax.Array, "Q 2"]:
        return self.weights.qpos_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    @property
    def skin_weights(self) -> Float[jax.Array, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[jax.Array, "V 3"]:
        params = self.get_rest_pose(batch_dims=())
        return self.forward_vertices(
            body_pose=params["body_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )

    def forward_skeleton(
        self,
        body_pose: Float[jax.Array, "B 23 N"] | Float[jax.Array, "B 23 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_vertices(
        self,
        body_pose: Float[jax.Array, "B 23 N"] | Float[jax.Array, "B 23 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[jax.Array, "B V 3"]:
        return backend.forward_vertices(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        body_pose: Float[jax.Array, "B 23 N"] | Float[jax.Array, "B 23 3 3"],
        global_translation: Float[jax.Array, "B 3"] | None = None,
        *,
        global_rotation: Float[jax.Array, "B N"] | Float[jax.Array, "B 3 3"] | None = None,
    ) -> Float[jax.Array, "B 24 4 4"]:
        return backend.forward_links(
            self.weights,
            self._qpos_order_body_pose(body_pose),
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_dims: tuple[int, ...] = (), dtype=jnp.float32) -> dict[str, jax.Array]:
        pose_ref = jnp.zeros((*batch_dims, len(BODY_JOINTS), 3), dtype=dtype)
        global_ref = jnp.zeros((*batch_dims, 3), dtype=dtype)
        return {
            "body_pose": SO3.identity_as(
                pose_ref,
                batch_dims=(*batch_dims, len(BODY_JOINTS)),
                rotation_type=self.rotation_type,
                xp=jnp,
            ),
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=batch_dims,
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=jnp,
            ),
            "global_translation": jnp.zeros((*batch_dims, 3), dtype=dtype),
        }

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        return self._preset_pose("t_pose", batch_dims=batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        return self._preset_pose("a_pose", batch_dims=batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...] = (), **kwargs) -> dict[str, jax.Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        axis_angle = jnp.asarray(SMPL_BODY_PRESETS[name], dtype=params["body_pose"].dtype)
        axis_angle = jnp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(axis_angle, src="axis_angle", dst=self.rotation_type, xp=jnp)
        return params

    def action_to_pd_target(self, action: Float[jax.Array, "... 69"]) -> Float[jax.Array, "... 69"]:
        return self.pd_action_offset.astype(action.dtype) + self.pd_action_scale.astype(action.dtype) * jnp.tanh(action)

    def joint_pos_to_action(self, joint_pos: Float[jax.Array, "... 69"]) -> Float[jax.Array, "... 69"]:
        scaled = jnp.clip(
            (joint_pos - self.pd_action_offset.astype(joint_pos.dtype)) / self.pd_action_scale.astype(joint_pos.dtype),
            -0.999,
            0.999,
        )
        return jnp.atanh(scaled)

    def to_qpos(
        self,
        body_pose: Float[jax.Array, "... 23 3"],
        global_translation: Float[jax.Array, "... 3"] | None = None,
        global_rotation: Float[jax.Array, "... 3"] | None = None,
    ) -> Float[jax.Array, "... 76"]:
        if self.rotation_type != "axis_angle":
            raise ValueError("to_qpos expects an axis_angle SmplHumanoid.")
        batch_shape = body_pose.shape[:-2]
        if global_translation is None:
            global_translation = jnp.zeros((*batch_shape, 3), dtype=body_pose.dtype)
        if global_rotation is None:
            global_rotation = jnp.zeros((*batch_shape, 3), dtype=body_pose.dtype)
        root_quat = SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=jnp)
        flat_body_pose = body_pose.reshape(-1, *body_pose.shape[-2:])
        ordered = [flat_body_pose[:, smpl_index] for _, smpl_index in BODY_JOINTS]
        joint_pos = jnp.stack(ordered, axis=1).reshape(*batch_shape, ACTION_SIZE)
        return jnp.concat([global_translation, root_quat, joint_pos], axis=-1).reshape(*batch_shape, QPOS_SIZE)

    def _qpos_order_body_pose(self, body_pose: jax.Array) -> jax.Array:
        axis = -3 if self.rotation_type in ("matrix", "rotmat") else -2
        return jnp.take(body_pose, jnp.asarray(BODY_POSE_ORDER), axis=axis)
