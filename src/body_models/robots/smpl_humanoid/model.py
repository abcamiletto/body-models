"""Single-source SMPL humanoid model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3
from trimesh import Trimesh

from body_models import common
from body_models.rigid import RigidBodyParameters, RigidModel
from body_models.robots.smpl_humanoid import core
from body_models.robots.smpl_humanoid.constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.robots.smpl_humanoid.io import load_model_data
from body_models.runtime import ArrayRuntime
from body_models.state import StateMaterializer

Array = Any


@dataclass(frozen=True)
class SmplHumanoidConfig:
    source: Path | str


class SmplHumanoidModel(RigidModel):
    """Backend-independent SMPL humanoid interface and orchestration."""

    JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self,
        source: Path | str = "humenv",
        *,
        runtime: ArrayRuntime,
        materialize: StateMaterializer,
    ) -> None:
        self._runtime = runtime
        self._config = SmplHumanoidConfig(source)
        self.weights = materialize(load_model_data(source))

    @property
    def actuated_joint_types(self) -> list[str]:
        return self.weights.actuated_joint_types

    def forward_skeleton(
        self,
        parameters: RigidBodyParameters,
        *,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Compute posed joint transforms."""
        weights = self.weights
        return core.forward_skeleton(
            local_offsets=weights.local_offsets,
            rest_local_rotations=weights.rest_local_rotations,
            actuated_joint_indices=weights.actuated_joint_indices,
            parents=weights.parents,
            body_pose=parameters.body_pose,
            global_translation=parameters.global_translation,
            global_rotation=parameters.global_rotation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def forward_links(
        self,
        parameters: RigidBodyParameters,
    ) -> Float[Array, "*batch L 4 4"]:
        """Compute posed link transforms."""
        skeleton = self.forward_skeleton(parameters)
        return self._link_transforms(skeleton)

    def forward_meshes(
        self,
        parameters: RigidBodyParameters,
    ) -> list[Trimesh]:
        """Build one posed render mesh per batch element."""
        links = self.forward_links(parameters)
        return self._meshes_from_links(links)

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> RigidBodyParameters:
        """Return zero humanoid pose controls."""
        return self._zero_body_parameters(batch_dims, dtype)

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> RigidBodyParameters:
        """Return the SMPL humanoid T-pose."""
        return self._preset_pose("t_pose", batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> RigidBodyParameters:
        """Return the SMPL humanoid A-pose."""
        return self._preset_pose("a_pose", batch_dims, **kwargs)

    def from_smpl_motion(
        self,
        smpl_body_pose: Float[Array, "*batch 23 3"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
    ) -> RigidBodyParameters:
        """Convert canonical SMPL motion into humanoid controls."""
        runtime = self._runtime
        xp = self._runtime.xp
        ordered = xp.stack([smpl_body_pose[..., index, :] for _, index in BODY_JOINTS], axis=-2)
        batch_shape = smpl_body_pose.shape[:-2]
        body_pose = SO3.conversions.from_axis_angle_to_euler(
            ordered,
            convention="XYZ",
            xp=xp,
        ).reshape(*batch_shape, self.num_actuated)
        if global_translation is None:
            global_translation = runtime.zeros((*batch_shape, 3), like=smpl_body_pose)
        if global_rotation is None:
            global_rotation = runtime.zeros((*batch_shape, 3), like=smpl_body_pose)
        elif pelvis_rotation is not None:
            root_rotation = global_rotation
            root_rotation = SO3.multiply(
                SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=xp),
                SO3.convert(pelvis_rotation, src="axis_angle", dst="quat", xp=xp),
                xp=xp,
            )
            global_rotation = SO3.convert(root_rotation, src="quat", dst="axis_angle", xp=xp)
        return RigidBodyParameters(body_pose, global_rotation, global_translation)

    def to_smpl_motion(self, qpos: Float[Array, "*batch Q"]) -> dict[str, Float[Array, "..."]]:
        """Convert MuJoCo qpos into canonical SMPL motion."""
        runtime = self._runtime
        xp = runtime.xp
        coord = runtime.asarray(self.mujoco_to_model, like=qpos)
        model_to_mujoco = coord.mT
        root_rotation_mujoco = SO3.conversions.from_quat_to_rotmat(
            qpos[..., 3:7],
            convention="wxyz",
            xp=xp,
        )
        root_rotation = coord @ root_rotation_mujoco @ model_to_mujoco
        ordered = SO3.conversions.from_euler_to_axis_angle(
            qpos[..., 7:].reshape(*qpos.shape[:-1], len(BODY_JOINTS), 3),
            convention="XYZ",
            xp=xp,
        )
        smpl_body_pose = runtime.zeros((*qpos.shape[:-1], 23, 3), like=qpos)
        for joint_index, (_, smpl_index) in enumerate(BODY_JOINTS):
            smpl_body_pose = common.set(
                smpl_body_pose,
                (..., smpl_index, slice(None)),
                ordered[..., joint_index, :],
                xp=xp,
            )
        return {
            "smpl_body_pose": smpl_body_pose,
            "global_translation": xp.squeeze(coord @ qpos[..., :3, None], axis=-1),
            "global_rotation": SO3.conversions.from_rotmat_to_axis_angle(root_rotation, xp=xp),
        }

    def _preset_pose(
        self,
        name: str,
        batch_dims: tuple[int, ...],
        **kwargs: Any,
    ) -> RigidBodyParameters:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        runtime = self._runtime
        xp = runtime.xp
        axis_angle = runtime.asarray(SMPL_BODY_PRESETS[name], like=params.body_pose)
        ordered = xp.stack([axis_angle[index] for _, index in BODY_JOINTS])
        ordered = SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=xp).reshape(-1)
        body_pose = xp.broadcast_to(ordered, (*batch_dims, ordered.shape[0]))
        return params._replace(body_pose=body_pose)


__all__ = ["SmplHumanoidConfig", "SmplHumanoidModel"]
