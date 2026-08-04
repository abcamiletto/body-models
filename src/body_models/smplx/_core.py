"""Backend-independent SMPL-X pose and identity preparation."""

from typing import Any, TypedDict

from jaxtyping import Float

from body_models import _smpl_family as family
from body_models._common import deformation
from body_models._rotations import RotationType

Array = Any
Front = tuple[list[int], list[int]]

SmplxSkeletonIdentity = deformation.SkeletonIdentity


class JointRegressor(TypedDict):
    """A vertex-to-joint mapping projected through the SMPL-X deformation bases."""

    weight_sums: Float[Array, "K"]
    skinning_weights: Float[Array, "K J"]
    template: Float[Array, "K J 3"]
    identity_directions: Float[Array, "K J 3 C"]
    pose_directions: Float[Array, "K J 3 P"]


prepare_identity = family.prepare_shape_expression_identity
prepare_skeleton_identity = family.prepare_shape_expression_skeleton_identity


def prepare_joint_regressor(
    mapping: Float[Array, "K V"],
    vertex_template: Float[Array, "V 3"],
    identity_directions: Float[Array, "V 3 C"],
    pose_directions: Float[Array, "P V*3"],
    skinning_weights: Float[Array, "V J"],
    *,
    xp: Any,
) -> JointRegressor:
    """Project a vertex-to-joint mapping through SMPL-X's linear bases once."""
    num_vertices = vertex_template.shape[0]
    pose_directions = xp.moveaxis(pose_directions.reshape(-1, num_vertices, 3), 0, -1)
    return {
        "weight_sums": xp.sum(mapping, axis=-1),
        "skinning_weights": mapping @ skinning_weights,
        "template": _project_vertex_values(mapping, skinning_weights, vertex_template, xp=xp),
        "identity_directions": _project_vertex_values(mapping, skinning_weights, identity_directions, xp=xp),
        "pose_directions": _project_vertex_values(mapping, skinning_weights, pose_directions, xp=xp),
    }


def _project_vertex_values(
    mapping: Float[Array, "K V"],
    skinning_weights: Float[Array, "V J"],
    values: Float[Array, "V *dims"],
    *,
    xp: Any,
) -> Float[Array, "K J *dims"]:
    """Project through only the nonzero skin weights to avoid a huge dense temporary."""
    value_shape = values.shape[1:]
    flat_values = values.reshape(values.shape[0], -1)
    projected = []
    for joint_index in range(skinning_weights.shape[1]):
        weights = skinning_weights[:, joint_index]
        vertices = xp.where(weights != 0)[0]
        weighted_mapping = mapping[:, vertices] * weights[vertices]
        projected.append(weighted_mapping @ flat_values[vertices])
    return xp.stack(projected, axis=1).reshape(mapping.shape[0], skinning_weights.shape[1], *value_shape)


def forward_joint_positions(
    regressor: JointRegressor,
    rest_points: Float[Array, "*batch K J 3"],
    pose: deformation.SkinningPose,
    *,
    xp: Any,
) -> Float[Array, "*batch K 3"]:
    """Evaluate posed joint positions without materializing the SMPL-X mesh."""
    offsets = xp.einsum("...p,kjdp->...kjd", pose["pose_coefficients"], regressor["pose_directions"])
    points = rest_points + offsets
    transforms = pose["skinning_transforms"]
    positions = xp.einsum("...jcd,...kjd->...kc", transforms[..., :3, :3], points)
    translations = xp.einsum(
        "kj,...jc->...kc",
        regressor["skinning_weights"],
        transforms[..., :3, 3],
    )
    return positions + translations


def _pose_matrices(
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    xp: Any,
) -> Float[Array, "*batch 55 3 3"]:
    hand_axis_angle = family.add_axis_angle_mean(
        hand_pose,
        hand_mean,
        rotation_type,
        xp=xp,
    )
    return family.assemble_pose_matrices(
        [
            (body_pose, rotation_type),
            (head_pose, rotation_type),
            (hand_axis_angle, "axis_angle"),
        ],
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )


def prepare_pose(
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    rest_joints: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> deformation.SkinningPose:
    """Prepare SMPL-X transforms and pose-corrective coefficients."""
    pose_matrices = _pose_matrices(
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )
    return family.prepare_pose(
        pose_matrices,
        kinematic_fronts,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        xp=xp,
    )


def prepare_skeleton(
    kinematic_fronts: list[Front],
    hand_mean: Float[Array, "2 45"],
    body_pose: Float[Array, "*batch 21 N"] | Float[Array, "*batch 21 3 3"],
    head_pose: Float[Array, "*batch 3 N"] | Float[Array, "*batch 3 3 3"],
    hand_pose: Float[Array, "*batch 30 N"] | Float[Array, "*batch 30 3 3"],
    pelvis_rotation: Float[Array, "*batch N"] | Float[Array, "*batch 3 3"] | None,
    rotation_type: RotationType,
    *,
    local_joint_offsets: Float[Array, "*identity_batch J 3"],
    xp: Any,
) -> Float[Array, "*batch J 4 4"]:
    """Prepare only posed SMPL-X joint transforms."""
    pose_matrices = _pose_matrices(
        hand_mean,
        body_pose,
        head_pose,
        hand_pose,
        pelvis_rotation,
        rotation_type,
        xp=xp,
    )
    return family.forward_skeleton(
        pose_matrices,
        local_joint_offsets,
        kinematic_fronts,
        xp=xp,
    )


__all__ = [
    "JointRegressor",
    "forward_joint_positions",
    "prepare_identity",
    "prepare_joint_regressor",
    "prepare_pose",
]
