"""SMPL humanoid model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from nanomanifold import SO3
from trimesh import Trimesh

from body_models import _common as common
from body_models._base import ParameterSpec, RigidBodyModel
from body_models._runtime import ArrayRuntime
from body_models.smpl_humanoid import _core as core
from body_models.smpl_humanoid._constants import BODY_JOINTS, SMPL_BODY_PRESETS, SMPL_HUMANOID_JOINTS
from body_models.smpl_humanoid._io import load_model_data

Array = Any


@dataclass(frozen=True)
class SmplHumanoidConfig:
    model_path: Path | str | None
    variant: Literal["humenv", "phc", "smplsim"]


class SmplHumanoid(RigidBodyModel):
    """Rigid SMPL-compatible humanoid loaded from MJCF."""

    _COMMON_JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        variant: Literal["humenv", "phc", "smplsim"] = "humenv",
        runtime: ArrayRuntime,
    ) -> None:
        runtime = self._set_runtime(runtime)
        self._config = SmplHumanoidConfig(model_path, variant)
        source = variant if model_path is None else model_path
        self._weights = runtime._materialize(load_model_data(source))

    @property
    def actuated_joint_types(self) -> list[str]:
        return self._weights.actuated_joint_types

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "body_pose": ParameterSpec((self.num_dofs,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch 24 4 4"]:
        """Compute posed joint transforms."""
        weights = self._weights
        return core.forward_skeleton(
            local_offsets=weights.local_offsets,
            rest_local_rotations=weights.rest_local_rotations,
            actuated_joint_indices=weights.actuated_joint_indices,
            parents=weights.parents,
            body_pose=body_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def forward_links(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch L 4 4"]:
        """Compute posed link transforms."""
        skeleton = self.forward_skeleton(
            body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return self._link_transforms(skeleton)

    def forward_meshes(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> list[Trimesh]:
        """Build one posed render mesh per batch element."""
        links = self.forward_links(
            body_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return self._meshes_from_links(links)

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL humanoid T-pose."""
        return self._preset_pose("t_pose", batch_dims, dtype)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL humanoid A-pose."""
        return self._preset_pose("a_pose", batch_dims, dtype)

    def parameters_from_smpl(
        self,
        smpl_body_pose: Float[Array, "*batch 23 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Convert canonical SMPL motion into humanoid controls."""
        xp = self._runtime.xp
        ordered = xp.stack([smpl_body_pose[..., index, :] for _, index in BODY_JOINTS], axis=-2)
        motion = {
            "body_pose": SO3.conversions.from_axis_angle_to_euler(
                ordered,
                convention="XYZ",
                xp=xp,
            ).reshape(*smpl_body_pose.shape[:-2], self.num_dofs)
        }
        if global_translation is not None:
            motion["global_translation"] = global_translation
        if global_rotation is not None:
            root_rotation = global_rotation
            if pelvis_rotation is not None:
                root_rotation = SO3.multiply(
                    SO3.convert(global_rotation, src="axis_angle", dst="quat", xp=xp),
                    SO3.convert(pelvis_rotation, src="axis_angle", dst="quat", xp=xp),
                    xp=xp,
                )
                root_rotation = SO3.convert(root_rotation, src="quat", dst="axis_angle", xp=xp)
            motion["global_rotation"] = root_rotation
        return motion

    def smpl_parameters_from_qpos(
        self,
        qpos: Float[Array, "*batch Q"],
    ) -> dict[str, Float[Array, "..."]]:
        """Convert MuJoCo qpos into canonical SMPL motion."""
        runtime = self._runtime
        xp = runtime.xp
        coord = runtime.asarray(self._mujoco_to_model(), like=qpos)
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
            smpl_body_pose = common.at_set(
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
        dtype: Any | None,
    ) -> dict[str, Float[Array, "..."]]:
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        runtime = self._runtime
        xp = runtime.xp
        axis_angle = runtime.asarray(SMPL_BODY_PRESETS[name], like=params["body_pose"])
        ordered = xp.stack([axis_angle[index] for _, index in BODY_JOINTS])
        ordered = SO3.conversions.from_axis_angle_to_euler(ordered, convention="XYZ", xp=xp).reshape(-1)
        params["body_pose"] = xp.broadcast_to(ordered, (*batch_dims, ordered.shape[0]))
        return params


__all__ = ["SmplHumanoid", "SmplHumanoidConfig"]
