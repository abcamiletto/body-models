"""NumPy backend for MANO model."""

from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int

from ..base import BodyModel
from nanomanifold import SO3

from ..rotations import VALID_ROTATION_TYPES
from . import core
from .io import compute_kinematic_fronts, get_joint_names, get_model_path, load_model_data, simplify_mesh

Array = np.ndarray

__all__ = ["MANO"]


class MANO(BodyModel):
    """MANO hand model with NumPy backend."""

    NUM_HAND_JOINTS = 15
    NUM_JOINTS = 16

    def __init__(
        self,
        model_path: Path | str | None = None,
        side: Literal["right", "left"] | None = None,
        flat_hand_mean: bool = False,
        simplify: float = 1.0,
        rotation_type: core.RotationType = "axis_angle",
    ):
        if side is not None and side not in ("right", "left"):
            raise ValueError(f"Invalid side: {side}. Must be 'right' or 'left'.")
        if rotation_type not in VALID_ROTATION_TYPES:
            raise ValueError(f"Invalid rotation_type: {rotation_type}")
        assert simplify >= 1.0

        self.side = side if side is not None else "right"
        self.rotation_type = rotation_type

        resolved_path = get_model_path(model_path, side)
        data = load_model_data(resolved_path)

        v_template_full = np.asarray(data["v_template"], dtype=np.float32)
        faces = np.asarray(data["f"], dtype=np.int32)
        lbs_weights = np.asarray(data["weights"], dtype=np.float32)
        shapedirs_full = np.asarray(data["shapedirs"], dtype=np.float32)
        shapedirs = shapedirs_full
        posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int64)
        parents[0] = -1

        if simplify > 1.0:
            target_faces = int(len(faces) / simplify)
            v_template, faces, vertex_map = simplify_mesh(v_template_full, faces, target_faces)
            lbs_weights = lbs_weights[vertex_map]
            shapedirs = shapedirs_full[vertex_map]
            posedirs = posedirs[vertex_map]
        else:
            v_template = v_template_full

        # Store arrays as instance attributes
        self.v_template = v_template
        self.v_template_full = v_template_full
        self.lbs_weights = lbs_weights
        self.J_regressor = J_regressor
        self.parents = parents.tolist()
        self._faces = faces

        hand_mean = np.asarray(data.get("hands_mean", np.zeros(45)), dtype=np.float32)
        if flat_hand_mean:
            hand_mean = np.zeros_like(hand_mean)
        self.hand_mean = hand_mean

        self.shapedirs = shapedirs
        self.shapedirs_full = shapedirs_full
        self.posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T

        self._kinematic_fronts = compute_kinematic_fronts(parents)
        self._joint_names = get_joint_names(data)

        # Precomputed joint regression matrices
        self._j_template = J_regressor @ v_template_full
        self._j_shapedirs = np.einsum("jv,vds->jds", J_regressor, shapedirs_full)

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._faces

    @property
    def num_joints(self) -> int:
        return self.NUM_JOINTS

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def num_vertices(self) -> int:
        return self.v_template.shape[0]

    @property
    def skin_weights(self) -> Float[Array, "V J"]:
        return self.lbs_weights

    @property
    def rest_vertices(self) -> Float[Array, "V 3"]:
        return self.v_template

    def forward_vertices(
        self,
        shape: Float[Array, "B|1 10"],
        hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
        wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
        global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
        global_translation: Float[Array, "B 3"] | None = None,
        vertex_indices=None,
    ) -> Float[Array, "B V 3"]:
        return core.forward_vertices(
            v_template=self.v_template,
            shapedirs=self.shapedirs,
            posedirs=self.posedirs,
            lbs_weights=self.lbs_weights,
            j_template=self._j_template,
            j_shapedirs=self._j_shapedirs,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            hand_mean=self.hand_mean,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=self.rotation_type,
        )

    def forward_skeleton(
        self,
        shape: Float[Array, "B|1 10"],
        hand_pose: Float[Array, "B 15 N"] | Float[Array, "B 15 3 3"],
        wrist_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
        global_rotation: Float[Array, "B N"] | Float[Array, "B 3 3"] | None = None,
        global_translation: Float[Array, "B 3"] | None = None,
        joint_indices=None,
    ) -> Float[Array, "B 16 4 4"]:
        return core.forward_skeleton(
            j_template=self._j_template,
            j_shapedirs=self._j_shapedirs,
            parents=self.parents,
            kinematic_fronts=self._kinematic_fronts,
            hand_mean=self.hand_mean,
            shape=shape,
            hand_pose=hand_pose,
            wrist_rotation=wrist_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            joint_indices=joint_indices,
            rotation_type=self.rotation_type,
        )

    def get_rest_pose(self, batch_size: int = 1, dtype=np.float32) -> dict[str, Array]:
        hand_pose_ref = np.zeros((batch_size, self.NUM_HAND_JOINTS, 3), dtype=dtype)
        wrist_ref = np.zeros((batch_size, 3), dtype=dtype)
        return {
            "shape": np.zeros((1, 10), dtype=dtype),
            "hand_pose": SO3.identity_as(
                hand_pose_ref,
                batch_dims=(batch_size, self.NUM_HAND_JOINTS),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "wrist_rotation": SO3.identity_as(
                wrist_ref,
                batch_dims=(batch_size,),
                rotation_type=self.rotation_type,
                xp=np,
            ),
            "global_translation": np.zeros((batch_size, 3), dtype=dtype),
        }
