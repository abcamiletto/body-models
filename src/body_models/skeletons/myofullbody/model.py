"""Single-source MyoFullBody model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float
from trimesh import Trimesh

from body_models.rigid import RigidModel
from body_models.runtime import ArrayRuntime
from body_models.state import StateMaterializer
from body_models.skeletons.myofullbody import core
from body_models.skeletons.myofullbody.constants import MYOFULLBODY_BODY_PRESETS, MYOFULLBODY_JOINTS
from body_models.skeletons.myofullbody.io import load_model_data

Array = Any


@dataclass(frozen=True)
class MyoFullBodyConfig:
    """Static MyoFullBody behavior outside array state."""


class MyoFullBodyModel(RigidModel):
    """Backend-independent MyoFullBody interface and orchestration."""

    JOINTS = MYOFULLBODY_JOINTS
    mujoco_to_model = core.MUJOCO_TO_KIMODO

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        runtime: ArrayRuntime,
        materialize: StateMaterializer,
    ) -> None:
        self._runtime = runtime
        self._config = MyoFullBodyConfig()
        self.weights = materialize(load_model_data(model_path))

    @property
    def actuated_joint_types(self) -> list[str]:
        return self.weights.actuated_joint_types

    @property
    def site_names(self) -> list[str]:
        return self.weights.site_names

    @property
    def site_positions(self) -> Float[Array, "S 3"]:
        return self.weights.site_positions

    @property
    def site_body_indices(self) -> list[int]:
        return self.weights.site_body_indices

    @property
    def tendons(self) -> list[dict]:
        return self.weights.tendons

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed body transforms."""
        weights = self.weights
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
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch L 4 4"]:
        """Compute posed link transforms."""
        skeleton = self.forward_skeleton(body_pose, global_translation, global_rotation=global_rotation)
        return self._link_transforms(skeleton)

    def forward_meshes(
        self,
        body_pose: Float[Array, "*batch Q"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
    ) -> list[Trimesh]:
        """Build one posed render mesh per batch element."""
        links = self.forward_links(body_pose, global_translation, global_rotation=global_rotation)
        return self._meshes_from_links(links)

    def world_sites(self, skeleton: Float[Array, "*batch J 4 4"]) -> Float[Array, "*batch S 3"]:
        """Transform body-local muscle sites with a prepared skeleton."""
        return core.world_sites(
            skeleton,
            self.weights.site_positions,
            self.weights.site_body_indices,
            xp=self._runtime.xp,
        )

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Return zero musculoskeletal controls."""
        return self._zero_pose("body_pose", batch_dims, dtype)

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Float[Array, "..."]]:
        """Return the MyoFullBody T-pose."""
        return self._preset_pose("t_pose", batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Float[Array, "..."]]:
        """Return the MyoFullBody A-pose."""
        return self._preset_pose("a_pose", batch_dims, **kwargs)

    def _preset_pose(
        self,
        name: str,
        batch_dims: tuple[int, ...],
        **kwargs: Any,
    ) -> dict[str, Float[Array, "..."]]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        pose = self._runtime.asarray(MYOFULLBODY_BODY_PRESETS[name], like=params["body_pose"])
        params["body_pose"] = self._runtime.xp.broadcast_to(pose, (*batch_dims, *pose.shape))
        return params


__all__ = ["MyoFullBodyConfig", "MyoFullBodyModel"]
