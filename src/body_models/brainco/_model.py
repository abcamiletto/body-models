"""BrainCo Revo 2 model implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jaxtyping import Float
from trimesh import Trimesh

from body_models._base import ParameterSpec, RigidBodyModel
from body_models._constants import Joint
from body_models._runtime import RuntimeLike
from body_models.brainco import _core as core
from body_models.brainco._constants import BRAINCO_HAND_PRESETS, LEFT_BRAINCO_JOINTS, RIGHT_BRAINCO_JOINTS
from body_models.brainco._io import Side, load_model_data

Array = Any


@dataclass(frozen=True)
class BrainCoConfig:
    side: Side


class BrainCoHand(RigidBodyModel):
    """Rigid articulated BrainCo Revo 2 hand."""

    has_hands = True

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        side: Side = "right",
        runtime: RuntimeLike = "numpy",
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"Invalid side: {side!r}")
        runtime = self._set_runtime(runtime)
        self._config = BrainCoConfig(side)
        self._weights = runtime.materialize(load_model_data(model_path, side=side))

    @property
    def side(self) -> Side:
        return self._config.side

    @property
    def common_joints(self) -> Mapping[Joint, str]:
        return LEFT_BRAINCO_JOINTS if self.side == "left" else RIGHT_BRAINCO_JOINTS

    @property
    def actuated_joint_types(self) -> list[str]:
        return ["hinge"] * self.num_actuated

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "hand_pose": ParameterSpec((self.num_actuated,), "pose"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def _mujoco_to_model(self):
        return core.MUJOCO_TO_KIMODO

    def forward_skeleton(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute posed joint transforms."""
        weights = self._weights
        return core.forward_skeleton(
            local_offsets=weights.local_offsets,
            rest_local_rotations=weights.rest_local_rotations,
            actuated_joint_axes=weights.actuated_joint_axes,
            actuated_joint_indices=weights.actuated_joint_indices,
            coupled_joint_axes=weights.coupled_joint_axes,
            coupled_joint_indices=weights.coupled_joint_indices,
            coupled_driver_indices=weights.coupled_driver_indices,
            coupled_polycoef=weights.coupled_polycoef,
            parents=weights.parents,
            pose=hand_pose,
            global_translation=global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def forward_links(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch L 4 4"]:
        """Compute posed link transforms."""
        skeleton = self.forward_skeleton(
            hand_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return self._link_transforms(skeleton)

    def forward_meshes(
        self,
        hand_pose: Float[Array, "*batch Q"],
        *,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> list[Trimesh]:
        """Build one posed render mesh per batch element."""
        links = self.forward_links(
            hand_pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return self._meshes_from_links(links)

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the configured default or canonical hand pose."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")
        params = super().get_rest_pose(batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            hand_pose = self._runtime.asarray(BRAINCO_HAND_PRESETS[self.side][hands], like=params["hand_pose"])
            params["hand_pose"] = self._runtime.xp.broadcast_to(hand_pose, (*batch_dims, self.num_actuated))
        return params


__all__ = ["BrainCoConfig", "BrainCoHand"]
