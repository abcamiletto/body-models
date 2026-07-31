"""Shape-dependent identities for the rigid SMPL-X mannequin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import numpy as np
from jaxtyping import Float

from body_models.smpl_humanoid import _io as io
from body_models.smplx import SMPLX

Array = Any


class SmplxMannequinIdentity(TypedDict):
    """Skeleton and rigid geometry prepared once for one beta vector."""

    rest_joints: Float[Array, "J 3"]
    local_joint_offsets: Float[Array, "J 3"]
    rest_vertices: NotRequired[Float[Array, "V 3"]]
    link_local_vertices: NotRequired[Float[Array, "V 3"]]


@dataclass(frozen=True)
class IdentityTemplate:
    local_offsets: np.ndarray
    rest_joints: np.ndarray
    link_local_vertices: np.ndarray
    joint_geom_anchors: tuple[int, ...]


def build_template(weights: io.SmplHumanoidWeights) -> IdentityTemplate:
    """Convert the neutral asset geometry into its owning joints' frames."""
    rest_joints = _joints_from_offsets(weights.local_offsets, weights.parents)
    local_vertices = np.empty_like(weights.vertices)
    anchors = list(weights.link_joint_indices)

    for link, (start, count) in enumerate(zip(weights.link_vertex_starts, weights.link_vertex_counts, strict=True)):
        vertices = weights.vertices[start : start + count]
        rotation = weights.link_geom_rotations[link]
        position = weights.link_geom_positions[link]
        local_vertices[start : start + count] = vertices @ rotation.T + position

        name = weights.link_names[link]
        if "shoulder_ball_L" in name:
            anchors[link] = weights.joint_names.index("L_Shoulder")
        elif "shoulder_ball_R" in name:
            anchors[link] = weights.joint_names.index("R_Shoulder")

    return IdentityTemplate(
        local_offsets=np.asarray(weights.local_offsets),
        rest_joints=rest_joints,
        link_local_vertices=local_vertices,
        joint_geom_anchors=tuple(anchors),
    )


def prepare(
    weights: io.SmplHumanoidWeights,
    template: IdentityTemplate,
    smplx: SMPLX | None,
    shape: np.ndarray,
    *,
    skip_vertices: bool,
) -> SmplxMannequinIdentity:
    """Prepare a symmetric, length-only identity from one SMPL-X beta vector."""
    shape = np.asarray(shape)
    if shape.ndim != 1:
        raise ValueError(f"shape must have shape [S], got {shape.shape}.")

    local_offsets = template.local_offsets.copy()
    if np.any(shape):
        if smplx is None:
            raise ValueError("A SMPL-X model is required for nonzero shape coefficients.")
        expression = np.zeros(10, dtype=shape.dtype)
        smplx_identity = smplx.prepare_identity(shape, expression)
        measured_offsets = _mapped_smplx_offsets(weights, smplx, smplx_identity["rest_joints"])
        local_offsets = _symmetric_length_offsets(template.local_offsets, measured_offsets, weights.joint_names)

    rest_joints = _joints_from_offsets(local_offsets, weights.parents)
    identity: SmplxMannequinIdentity = {
        "rest_joints": rest_joints,
        "local_joint_offsets": local_offsets,
    }
    if skip_vertices:
        return identity

    local_vertices, rest_vertices = _shape_vertices(weights, template, local_offsets, rest_joints)
    identity["link_local_vertices"] = local_vertices
    identity["rest_vertices"] = rest_vertices
    return identity


def _mapped_smplx_offsets(
    weights: io.SmplHumanoidWeights,
    smplx: SMPLX,
    smplx_joints: np.ndarray,
) -> np.ndarray:
    renamed = {
        "Torso": "Spine1",
        "Spine": "Spine2",
        "Chest": "Spine3",
        "L_Toe": "L_Foot",
        "R_Toe": "R_Foot",
        "L_Thorax": "L_Collar",
        "R_Thorax": "R_Collar",
        "L_Hand": "L_Wrist",
        "R_Hand": "R_Wrist",
    }
    indices = [smplx.joint_names.index(renamed.get(name, name)) for name in weights.joint_names]
    joints = np.asarray(smplx_joints)[indices] - smplx_joints[0]
    offsets = np.zeros_like(joints)
    for joint in range(1, len(weights.parents)):
        offsets[joint] = joints[joint] - joints[weights.parents[joint]]
    return offsets


def _symmetric_length_offsets(
    neutral_offsets: np.ndarray,
    measured_offsets: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    result = neutral_offsets.copy()
    indices = {name: index for index, name in enumerate(joint_names)}
    paired = set()
    reflection = np.array((-1.0, 1.0, 1.0), dtype=result.dtype)

    for name, left in indices.items():
        if not name.startswith("L_"):
            continue
        right = indices[f"R_{name[2:]}"]
        paired.update((left, right))
        neutral_length = np.linalg.norm(neutral_offsets[left])
        if neutral_length == 0.0:
            result[[left, right]] = 0.0
            continue
        length = 0.5 * (np.linalg.norm(measured_offsets[left]) + np.linalg.norm(measured_offsets[right]))
        result[left] *= length / neutral_length
        result[right] = result[left] * reflection

    for joint in range(1, len(result)):
        if joint in paired:
            continue
        neutral_length = np.linalg.norm(neutral_offsets[joint])
        if neutral_length > 0.0:
            result[joint] *= np.linalg.norm(measured_offsets[joint]) / neutral_length
    return result


def _shape_vertices(
    weights: io.SmplHumanoidWeights,
    template: IdentityTemplate,
    local_offsets: np.ndarray,
    rest_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = _joint_transforms(template.local_offsets, local_offsets, weights.parents)
    local_parts = []
    rest_parts = []
    for link, (owner, start, count, name) in enumerate(
        zip(
            weights.link_joint_indices,
            weights.link_vertex_starts,
            weights.link_vertex_counts,
            weights.link_names,
            strict=True,
        )
    ):
        vertices = template.link_local_vertices[start : start + count]
        if "__joint_" in name:
            anchor = template.joint_geom_anchors[link]
            neutral_anchor = template.rest_joints[anchor] - template.rest_joints[owner]
            shaped_anchor = rest_joints[anchor] - rest_joints[owner]
            vertices = vertices + shaped_anchor - neutral_anchor
        else:
            vertices = vertices @ transforms[owner].T
        local_parts.append(vertices)
        rest_parts.append(vertices + rest_joints[owner])
    return np.concatenate(local_parts), np.concatenate(rest_parts)


def _joint_transforms(
    neutral_offsets: np.ndarray,
    target_offsets: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    transforms = np.repeat(np.eye(3, dtype=target_offsets.dtype)[None], len(parents), axis=0)
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)

    for joint, joint_children in enumerate(children):
        if joint_children:
            neutral_axes = neutral_offsets[joint_children].T
            target_axes = target_offsets[joint_children].T
            transforms[joint] += (target_axes - neutral_axes) @ np.linalg.pinv(neutral_axes)
    return transforms


def _joints_from_offsets(offsets: np.ndarray, parents: list[int]) -> np.ndarray:
    joints = np.zeros_like(offsets)
    for joint in range(1, len(parents)):
        joints[joint] = joints[parents[joint]] + offsets[joint]
    return joints


__all__ = ["IdentityTemplate", "SmplxMannequinIdentity", "build_template", "prepare"]
