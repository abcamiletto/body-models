"""MyoFullBody model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float
from trimesh import Trimesh

from body_models._base import ParameterSpec, RigidBodyModel
from body_models._runtime import ArrayRuntime
from body_models.myofullbody import _constants as constants
from body_models.myofullbody import _core as core
from body_models.myofullbody._io import load_model_data

Array = Any


@dataclass(frozen=True)
class MyoFullBodyConfig:
    """Static MyoFullBody behavior outside array state."""


class MyoFullBody(RigidBodyModel):
    """Rigid articulated musculoskeletal full-body model."""

    _COMMON_JOINTS = constants.MYOFULLBODY_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        runtime: ArrayRuntime,
    ) -> None:
        self._attach_runtime(runtime)
        self._config = MyoFullBodyConfig()
        self._weights = runtime._materialize(load_model_data(model_path))

    @property
    def actuated_joint_types(self) -> list[str]:
        return self._weights.actuated_joint_types

    @property
    def _pose_control_joints(self) -> tuple[tuple[int, ...], ...]:
        return tuple((joint,) for joint, count in enumerate(self._weights.body_actuated_counts) for _ in range(count))

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "body_pose": ParameterSpec((self.num_dofs,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def _mujoco_to_model(self):
        return constants.MUJOCO_TO_MYOFULLBODY

    @property
    def site_names(self) -> list[str]:
        return self._weights.site_names

    @property
    def site_positions(self) -> Float[Array, "S 3"]:
        return self._weights.site_positions

    @property
    def site_body_indices(self) -> list[int]:
        return self._weights.site_body_indices

    @property
    def tendons(self) -> list[dict]:
        return self._weights.tendons

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed body transforms."""
        weights = self._weights
        return core.forward_skeleton(
            local_offsets=weights.local_offsets,
            rest_local_rotations=weights.rest_local_rotations,
            parents=weights.parents,
            body_actuated_starts=weights.body_actuated_starts,
            body_actuated_counts=weights.body_actuated_counts,
            actuated_joint_axes=weights.actuated_joint_axes,
            actuated_joint_anchors=weights.actuated_joint_anchors,
            hinge_mask=weights.hinge_mask,
            slide_mask=weights.slide_mask,
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

    def world_sites(self, skeleton: Float[Array, "*batch J 4 4"]) -> Float[Array, "*batch S 3"]:
        """Transform body-local muscle sites with a prepared skeleton."""
        return core.world_sites(
            skeleton,
            self._weights.site_positions,
            self._weights.site_body_indices,
            xp=self._runtime.xp,
        )

    def parameters_from_qpos(
        self,
        qpos: Float[Array, "*batch Q"],
    ) -> dict[str, Float[Array, "..."]]:
        """Convert MuJoCo qpos into model forward parameters."""
        return core.parameters_from_qpos(qpos, xp=self._runtime.xp)

    def get_tpose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MyoFullBody T-pose."""
        return self._preset_pose("t_pose", batch_dims, dtype)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return the MyoFullBody A-pose."""
        return self._preset_pose("a_pose", batch_dims, dtype)

    def _preset_pose(
        self,
        name: str,
        batch_dims: tuple[int, ...],
        dtype: Any | None,
    ) -> dict[str, Float[Array, "..."]]:
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        pose = self._runtime.asarray(constants.MYOFULLBODY_BODY_PRESETS[name], like=params["body_pose"])
        params["body_pose"] = self._runtime.xp.broadcast_to(pose, (*batch_dims, *pose.shape))
        return params


__all__ = ["MyoFullBody", "MyoFullBodyConfig"]
