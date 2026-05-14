"""NumPy backend for the BrainCo Revo 2 robotic hand model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from body_models.base import BodyModel
from body_models.brainco.backends import core
from body_models.brainco.backends import numpy as backend
from body_models.brainco.io import Side, load_model_data
from body_models.brainco.constants import LEFT_BRAINCO_JOINTS, RIGHT_BRAINCO_JOINTS

__all__ = ["BrainCoHand"]


class BrainCoHand(BodyModel):
    """BrainCo Revo 2 as rigid STL links attached to its MuJoCo hand skeleton."""

    has_hands = True

    is_rigid_body = True

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        side: Side = "right",
        rotation_type: core.RotationType = "rotmat",
    ) -> None:
        if rotation_type not in core.VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        self.rotation_type = rotation_type
        self.num_rot_dims = 2 if rotation_type in ("matrix", "rotmat") else 1
        self.weights = load_model_data(model_path, side=side)

    @property
    def faces(self) -> Int[np.ndarray, "F 3"]:
        return self.weights.faces

    @property
    def side(self) -> Side:
        return self.weights.side

    @property
    def num_joints(self) -> int:
        return len(self.weights.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self.weights.joint_names

    @property
    def common_joints(self):
        return LEFT_BRAINCO_JOINTS if self.side == "left" else RIGHT_BRAINCO_JOINTS

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
    def qpos_joint_axes(self) -> Float[np.ndarray, "Q 3"]:
        return self.weights.qpos_joint_axes

    @property
    def qpos_joint_limits(self) -> Float[np.ndarray, "Q 2"]:
        return self.weights.qpos_joint_limits

    @property
    def link_names(self) -> list[str]:
        return self.weights.link_names

    @property
    def link_joint_indices(self) -> list[int]:
        return self.weights.link_joint_indices

    @property
    def link_vertex_starts(self) -> list[int]:
        return self.weights.link_vertex_starts

    @property
    def link_vertex_counts(self) -> list[int]:
        return self.weights.link_vertex_counts

    @property
    def link_face_starts(self) -> list[int]:
        return self.weights.link_face_starts

    @property
    def link_face_counts(self) -> list[int]:
        return self.weights.link_face_counts

    @property
    def num_vertices(self) -> int:
        return self.weights.vertices.shape[0]

    @property
    def skin_weights(self) -> Float[np.ndarray, "V J"]:
        raise NotImplementedError(core.SKIN_WEIGHTS_ERROR)

    @property
    def rest_vertices(self) -> Float[np.ndarray, "V 3"]:
        params = self.get_rest_pose(batch_size=1)
        return self.forward_vertices(
            hand_pose=params["hand_pose"],
            global_translation=params["global_translation"],
            global_rotation=params["global_rotation"],
        )[0]

    def forward_skeleton(
        self,
        hand_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        joint_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B J 4 4"]:
        return backend.forward_skeleton(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def forward_vertices(
        self,
        hand_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
        vertex_indices: list[int] | None = None,
    ) -> Float[np.ndarray, "B V 3"]:
        return backend.forward_vertices(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_links(
        self,
        hand_pose: Float[np.ndarray, "B Q N"] | Float[np.ndarray, "B Q 3 3"],
        global_translation: Float[np.ndarray, "B 3"] | None = None,
        *,
        global_rotation: Float[np.ndarray, "B N"] | Float[np.ndarray, "B 3 3"] | None = None,
    ) -> Float[np.ndarray, "B L 4 4"]:
        return backend.forward_links(
            self.weights,
            hand_pose,
            global_translation,
            global_rotation=global_rotation,
            rotation_type=self.rotation_type,
        )

    def link_mesh(self, link_name: str) -> dict[str, np.ndarray | str]:
        return core.link_mesh(
            self.weights.vertices,
            self.weights.faces,
            self.weights.link_vertex_starts,
            self.weights.link_vertex_counts,
            self.weights.link_face_starts,
            self.weights.link_face_counts,
            self.weights.link_names,
            link_name,
        )

    def get_rest_pose(
        self,
        batch_size: int = 1,
        dtype=np.float32,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}. Expected 'default', 'flat', or 'rest'.")

        pose_ref = np.zeros((batch_size, len(self.weights.qpos_joint_indices), 3), dtype=dtype)
        global_ref = np.zeros((batch_size, 3), dtype=dtype)
        hand_pose = SO3.identity_as(
            pose_ref,
            batch_dims=(batch_size, len(self.weights.qpos_joint_indices)),
            rotation_type=self.rotation_type,
            xp=np,
        )
        if hands == "rest":
            hinge_pose = np.full((batch_size, len(self.weights.qpos_joint_indices), 1), 0.65, dtype=dtype)
            hand_pose = SO3.convert(
                hinge_pose,
                src="hinge",
                dst=self.rotation_type,
                src_kwargs={"axes": self.weights.qpos_joint_axes},
                xp=np,
            )
        return {
            "hand_pose": hand_pose,
            "global_rotation": SO3.identity_as(
                global_ref,
                batch_dims=(batch_size,),
                rotation_type=core.GLOBAL_ROTATION_TYPES[self.rotation_type],
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
