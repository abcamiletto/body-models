"""SMPL humanoid model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3
from trimesh import Trimesh

from body_models import _common as common
from body_models._base import ParameterSpec, RigidBodyModel
from body_models._runtime import RuntimeLike
from body_models.smpl_humanoid import _core as core
from body_models.smpl_humanoid import identity as identity_ops
from body_models.smpl_humanoid._constants import (
    BODY_JOINTS,
    SMPL_BODY_PRESETS,
    SMPL_HUMANOID_JOINTS,
)
from body_models.smpl_humanoid._io import load_model_data
from body_models.smplx import SMPLX
from body_models.smplx._constants import SMPLX_BODY_PRESETS, SMPLX_HAND_PRESETS

Array = Any


@dataclass(frozen=True)
class SmplHumanoidConfig:
    model_path: Path | str | None
    variant: Literal[
        "mannequin",
        "mannequin_lod1",
        "mannequin_lod2",
        "humenv",
        "phc",
        "smplsim",
    ]


class SmplHumanoid(RigidBodyModel):
    """Rigid SMPL-compatible humanoid loaded from MJCF."""

    _COMMON_JOINTS = SMPL_HUMANOID_JOINTS

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        variant: Literal[
            "mannequin",
            "mannequin_lod1",
            "mannequin_lod2",
            "humenv",
            "phc",
            "smplsim",
        ] = "humenv",
        runtime: RuntimeLike = "numpy",
    ) -> None:
        runtime = self._set_runtime(runtime)
        self._config = SmplHumanoidConfig(model_path, variant)
        source = variant if model_path is None else model_path
        self._weights = runtime._materialize(load_model_data(source))

    @property
    def actuated_joint_types(self) -> list[str]:
        return self._weights.actuated_joint_types

    @property
    def link_vertex_starts(self) -> list[int]:
        return list(self._weights.link_vertex_starts)

    @property
    def link_vertex_counts(self) -> list[int]:
        return list(self._weights.link_vertex_counts)

    @property
    def link_face_starts(self) -> list[int]:
        return list(self._weights.link_face_starts)

    @property
    def link_face_counts(self) -> list[int]:
        return list(self._weights.link_face_counts)

    @property
    def link_geom_positions(self) -> Float[Array, "L 3"]:
        return self._weights.link_geom_positions

    @property
    def link_geom_rotations(self) -> Float[Array, "L 3 3"]:
        return self._weights.link_geom_rotations

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
        left_hand_pose: Float[Array, "*batch 15 3"] | None = None,
        right_hand_pose: Float[Array, "*batch 15 3"] | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        """Convert canonical SMPL motion into humanoid controls."""
        xp = self._runtime.xp
        if smpl_body_pose.shape[-2] == 21:
            padding = self._runtime.zeros((*smpl_body_pose.shape[:-2], 2, 3), like=smpl_body_pose)
            smpl_body_pose = xp.concat((smpl_body_pose, padding), axis=-2)
        if smpl_body_pose.shape[-2] != 23:
            raise ValueError(f"smpl_body_pose must contain 21 or 23 joints, got {smpl_body_pose.shape[-2]}")
        ordered = xp.stack([smpl_body_pose[..., index, :] for _, index in BODY_JOINTS], axis=-2)
        rotations = [ordered]
        has_fingers = self.num_dofs > len(BODY_JOINTS) * 3
        if has_fingers:
            hand_shape = (*smpl_body_pose.shape[:-2], 15, 3)
            if left_hand_pose is None:
                left_hand_pose = self._runtime.zeros(hand_shape, like=smpl_body_pose)
            if right_hand_pose is None:
                right_hand_pose = self._runtime.zeros(hand_shape, like=smpl_body_pose)
            rotations.extend((left_hand_pose, right_hand_pose))
        elif left_hand_pose is not None or right_hand_pose is not None:
            raise ValueError("This humanoid asset has no articulated finger joints.")
        motion = {
            "body_pose": SO3.conversions.from_axis_angle_to_euler(
                xp.concat(rotations, axis=-2),
                convention="XYZ",
                xp=xp,
            ).reshape(*smpl_body_pose.shape[:-2], self.num_dofs)
        }
        if global_translation is not None:
            motion["global_translation"] = global_translation
        if global_rotation is not None or pelvis_rotation is not None:
            root_rotation = global_rotation
            if root_rotation is None:
                assert pelvis_rotation is not None
                root_rotation = self._runtime.zeros(pelvis_rotation.shape, like=pelvis_rotation)
            if pelvis_rotation is not None:
                root_rotation = SO3.multiply(
                    SO3.convert(root_rotation, src="axis_angle", dst="quat", xp=xp),
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
        all_ordered = SO3.conversions.from_euler_to_axis_angle(
            qpos[..., 7:].reshape(*qpos.shape[:-1], self.num_dofs // 3, 3),
            convention="XYZ",
            xp=xp,
        )
        ordered = all_ordered[..., : len(BODY_JOINTS), :]
        smpl_body_pose = runtime.zeros((*qpos.shape[:-1], 23, 3), like=qpos)
        for joint_index, (_, smpl_index) in enumerate(BODY_JOINTS):
            smpl_body_pose = common.at_set(
                smpl_body_pose,
                (..., smpl_index, slice(None)),
                ordered[..., joint_index, :],
                xp=xp,
            )
        motion = {
            "smpl_body_pose": smpl_body_pose,
            "global_translation": xp.squeeze(coord @ qpos[..., :3, None], axis=-1),
            "global_rotation": SO3.conversions.from_rotmat_to_axis_angle(root_rotation, xp=xp),
        }
        if all_ordered.shape[-2] > len(BODY_JOINTS):
            fingers = all_ordered[..., len(BODY_JOINTS) :, :]
            motion["left_hand_pose"] = fingers[..., :15, :]
            motion["right_hand_pose"] = fingers[..., 15:, :]
        return motion

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


class SmplMannequin(SmplHumanoid):
    """Rigid, non-skinned mannequin with the canonical SMPL hierarchy."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        variant: Literal["mannequin", "mannequin_lod1", "mannequin_lod2"] = "mannequin",
        runtime: RuntimeLike = "numpy",
    ) -> None:
        super().__init__(model_path=model_path, variant=variant, runtime=runtime)


class SmplxMannequin(SmplMannequin):
    """Rigid mannequin with SMPL-X pose and shape parameters.

    Shape changes alter symmetric bone lengths and reposition rigid parts; they
    never change link thickness or introduce skinning.
    """

    has_hands = True
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    NUM_SHAPE_COEFFS = 10
    NUM_EXPR_COEFFS = 10

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        variant: Literal["mannequin", "mannequin_lod1", "mannequin_lod2"] = "mannequin",
        smplx_model: SMPLX | Path | str | None = None,
        runtime: RuntimeLike = "numpy",
    ) -> None:
        runtime = self._set_runtime(runtime)
        self._config = SmplHumanoidConfig(model_path, variant)
        weights = load_model_data(variant if model_path is None else model_path)
        self._identity_template = identity_ops.build_template(weights)
        self._weights = runtime._materialize(weights)
        if isinstance(smplx_model, Path | str):
            smplx_model = SMPLX(
                model_path=smplx_model,
                gender="neutral",
                flat_hand_mean=True,
                runtime="numpy",
            )
        self._smplx_model = smplx_model

    @property
    def parameter_spec(self) -> dict[str, ParameterSpec]:
        return {
            "shape": ParameterSpec((self.NUM_SHAPE_COEFFS,), "identity"),
            "expression": ParameterSpec((self.NUM_EXPR_COEFFS,), "identity"),
            "body_pose": ParameterSpec.rotation("axis_angle", count=self.NUM_BODY_JOINTS),
            "head_pose": ParameterSpec.rotation("axis_angle", count=self.NUM_HEAD_JOINTS),
            "hand_pose": ParameterSpec.rotation("axis_angle", count=self.NUM_HAND_JOINTS),
            "pelvis_rotation": ParameterSpec.rotation("axis_angle"),
            "global_rotation": ParameterSpec.rotation("axis_angle", role="transform"),
            "global_translation": ParameterSpec((3,), "transform"),
        }

    def prepare_identity(
        self,
        shape: Float[Array, "10"],
        expression: Float[Array, "10"] | None = None,
        *,
        skip_vertices: bool = False,
    ) -> identity_ops.SmplxMannequinIdentity:
        """Prepare one reusable, length-only identity from an unbatched beta vector."""
        del expression
        shape_np = self._runtime.to_numpy(shape)
        if np.any(shape_np) and self._smplx_model is None:
            self._smplx_model = SMPLX(gender="neutral", flat_hand_mean=True, runtime="numpy")
        identity = identity_ops.prepare(
            self._numpy_weights(),
            self._identity_template,
            self._smplx_model,
            shape_np,
            skip_vertices=skip_vertices,
        )
        return self._runtime._materialize(identity)

    def prepare_pose(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        head_pose: Float[Array, "*batch 3 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        identity: identity_ops.SmplxMannequinIdentity,
    ) -> dict[str, Float[Array, "..."]]:
        """Convert SMPL-X rotations to the mannequin's Euler coordinates."""
        del head_pose, identity
        return self._parameters_from_smplx(body_pose, hand_pose, pelvis_rotation=pelvis_rotation)

    def forward_skeleton(  # ty: ignore[invalid-method-override]
        self,
        body_pose: Float[Array, "*batch 21 3"],
        head_pose: Float[Array, "*batch 3 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        shape: Float[Array, "10"] | None = None,
        expression: Float[Array, "10"] | None = None,
        identity: identity_ops.SmplxMannequinIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute shape-adjusted mannequin joint transforms."""
        identity = self._resolve_identity(identity, shape, expression, skip_vertices=True)
        params = self._parameters_from_smplx(
            body_pose,
            hand_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return core.forward_skeleton(
            local_offsets=identity["local_joint_offsets"],
            rest_local_rotations=self._weights.rest_local_rotations,
            actuated_joint_indices=self._weights.actuated_joint_indices,
            parents=self._weights.parents,
            body_pose=params["body_pose"],
            global_rotation=params.get("global_rotation"),
            global_translation=params.get("global_translation"),
            joint_indices=joint_indices,
            xp=self._runtime.xp,
        )

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        head_pose: Float[Array, "*batch 3 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        shape: Float[Array, "10"] | None = None,
        expression: Float[Array, "10"] | None = None,
        identity: identity_ops.SmplxMannequinIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
        vertex_indices: Any | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute vertices for the shape-adjusted rigid parts."""
        identity = self._resolve_identity(identity, shape, expression, skip_vertices=False)
        skeleton = self.forward_skeleton(
            body_pose,
            head_pose,
            hand_pose,
            pelvis_rotation=pelvis_rotation,
            identity=identity,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        vertices = self._vertices_from_skeleton(skeleton, identity["link_local_vertices"])
        return vertices if vertex_indices is None else vertices[..., vertex_indices, :]

    def forward_links(self, *args: Any, **kwargs: Any) -> Float[Array, "*batch L 4 4"]:
        """Compute link transforms from SMPL-X parameters."""
        return self._link_transforms(self.forward_skeleton(*args, **kwargs))

    def forward_meshes(self, *args: Any, **kwargs: Any) -> list[Trimesh]:
        """Build one posed mannequin mesh per batch element."""
        vertices = self._runtime.to_numpy(self.forward_vertices(*args, **kwargs))
        if vertices.ndim == 2:
            vertices = vertices[None]
        vertices = vertices.reshape(-1, vertices.shape[-2], 3)
        faces = self._runtime.to_numpy(self.faces)
        return [Trimesh(vertices=item, faces=faces, process=False) for item in vertices]

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X rest pose with configurable hand means."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")
        params = RigidBodyModel.get_rest_pose(self, batch_dims=batch_dims, dtype=dtype)
        if hands != "default":
            hand_pose = self._runtime.asarray(SMPLX_HAND_PRESETS[hands], like=params["hand_pose"])
            hand_pose = hand_pose.reshape(self.NUM_HAND_JOINTS, 3)
            params["hand_pose"] = self._runtime.xp.broadcast_to(hand_pose, (*batch_dims, *hand_pose.shape))
        return params

    def get_tpose(self, **kwargs: Any) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X T-pose."""
        return self.get_rest_pose(**kwargs)

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        body_pose = self._runtime.asarray(SMPLX_BODY_PRESETS["a_pose"], like=params["body_pose"])
        params["body_pose"] = self._runtime.xp.broadcast_to(body_pose, (*batch_dims, *body_pose.shape))
        return params

    def _parameters_from_smplx(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> dict[str, Float[Array, "..."]]:
        if body_pose.shape[-2:] != (self.NUM_BODY_JOINTS, 3):
            raise ValueError(f"body_pose must have shape [..., 21, 3], got {tuple(body_pose.shape)}")
        if hand_pose.shape[-2:] != (self.NUM_HAND_JOINTS, 3):
            raise ValueError(f"hand_pose must have shape [..., 30, 3], got {tuple(hand_pose.shape)}")
        return self.parameters_from_smpl(
            body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            left_hand_pose=hand_pose[..., :15, :],
            right_hand_pose=hand_pose[..., 15:, :],
        )

    def _resolve_identity(
        self,
        identity: identity_ops.SmplxMannequinIdentity | None,
        shape: Float[Array, "10"] | None,
        expression: Float[Array, "10"] | None,
        *,
        skip_vertices: bool,
    ) -> identity_ops.SmplxMannequinIdentity:
        if identity is not None:
            conflicts = [name for name, value in (("shape", shape), ("expression", expression)) if value is not None]
            if conflicts:
                raise ValueError(f"identity cannot be combined with raw identity parameters: {', '.join(conflicts)}")
            return identity
        if shape is None:
            shape = self._runtime.zeros((self.NUM_SHAPE_COEFFS,), like=self._weights.vertices)
        return self.prepare_identity(shape, expression, skip_vertices=skip_vertices)

    def _vertices_from_skeleton(
        self,
        skeleton: Float[Array, "*batch J 4 4"],
        local_vertices: Float[Array, "V 3"],
    ) -> Float[Array, "*batch V 3"]:
        xp = self._runtime.xp
        parts = []
        for owner, start, count in zip(
            self._weights.link_joint_indices,
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            strict=True,
        ):
            rotation = skeleton[..., owner, :3, :3]
            translation = skeleton[..., owner, :3, 3]
            local = local_vertices[start : start + count]
            parts.append(xp.squeeze(rotation[..., None, :, :] @ local[..., None], axis=-1) + translation[..., None, :])
        return xp.concat(parts, axis=-2)

    def _numpy_weights(self):
        if self._runtime.name == "numpy":
            return self._weights
        return load_model_data(self._config.variant if self._config.model_path is None else self._config.model_path)


__all__ = ["SmplHumanoid", "SmplHumanoidConfig", "SmplMannequin", "SmplxMannequin"]
