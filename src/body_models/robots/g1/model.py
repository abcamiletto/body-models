"""Single-source Unitree G1 model program."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jaxtyping import Float
from nanomanifold import SO3
from trimesh import Trimesh

from body_models.rigid import RigidModel
from body_models.robots.g1 import core
from body_models.robots.g1.constants import G1_BODY_PRESETS, G1_JOINTS
from body_models.robots.g1.io import load_model_data
from body_models.runtime import Runtime

Array = Any


@dataclass(frozen=True)
class G1Config:
    convention: core.Convention


class G1Model(RigidModel):
    """Backend-independent Unitree G1 interface and orchestration."""

    JOINTS = G1_JOINTS

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        convention: core.Convention = "soma",
        runtime: Runtime,
    ) -> None:
        if convention not in ("soma", "mujoco"):
            raise ValueError(f"Invalid G1 convention: {convention!r}")
        self._runtime = runtime
        self._config = G1Config(convention)
        self.weights = runtime.convert_model_data(load_model_data(model_path, convention=convention))

    @property
    def convention(self) -> core.Convention:
        return self._config.convention

    @property
    def mujoco_to_model(self):
        if self.convention == "soma":
            return core.MUJOCO_TO_KIMODO
        return super().mujoco_to_model

    @property
    def actuated_joint_types(self) -> list[str]:
        return ["hinge"] * self.num_actuated

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch Q"],
        global_translation: Float[Array, "*batch 3"] | None = None,
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        weights = self.weights
        return core.forward_skeleton(
            local_offsets=weights.local_offsets,
            rest_local_rotations=weights.rest_local_rotations,
            actuated_joint_indices=weights.actuated_joint_indices,
            actuated_joint_axes=weights.actuated_joint_axes,
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

    def get_rest_pose(
        self,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
    ) -> dict[str, Array]:
        """Return zero G1 pose controls."""
        return self._zero_pose("body_pose", batch_dims, dtype)

    def get_tpose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Array]:
        """Return the G1 T-pose."""
        return self._preset_pose("t_pose", batch_dims, **kwargs)

    def get_apose(self, batch_dims: tuple[int, ...] = (), **kwargs: Any) -> dict[str, Array]:
        """Return the G1 A-pose."""
        return self._preset_pose("a_pose", batch_dims, **kwargs)

    def _preset_pose(self, name: str, batch_dims: tuple[int, ...], **kwargs: Any) -> dict[str, Array]:
        params = self.get_rest_pose(batch_dims=batch_dims, **kwargs)
        runtime = self._runtime
        axis_angle = runtime.asarray(G1_BODY_PRESETS[name], like=params["body_pose"])
        axis_angle = runtime.xp.broadcast_to(axis_angle, (*batch_dims, *axis_angle.shape))
        params["body_pose"] = SO3.convert(
            axis_angle,
            src="axis_angle",
            dst="hinge",
            dst_kwargs={"axes": self.weights.actuated_joint_axes},
            xp=runtime.xp,
        )[..., 0]
        return params


__all__ = ["G1Config", "G1Model"]
