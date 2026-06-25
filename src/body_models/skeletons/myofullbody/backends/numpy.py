"""NumPy backend kernels for MyoFullBody."""

import numpy as np
from jaxtyping import Float

from body_models.skeletons.myofullbody.backends import core
from body_models.skeletons.myofullbody.io import MyoFullBodyWeights


def forward_skeleton(
    weights: MyoFullBodyWeights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
    joint_indices=None,
):
    return core.forward_skeleton(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        parents=weights.parents,
        body_qpos_starts=weights.body_qpos_starts,
        body_qpos_counts=weights.body_qpos_counts,
        qpos_axes=weights.qpos_joint_axes,
        qpos_anchors=weights.qpos_joint_anchors,
        hinge_mask=weights.hinge_mask,
        slide_mask=weights.slide_mask,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        joint_indices=joint_indices,
        xp=np,
    )


def forward_links(
    weights: MyoFullBodyWeights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
):
    return core.forward_links(
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        parents=weights.parents,
        body_qpos_starts=weights.body_qpos_starts,
        body_qpos_counts=weights.body_qpos_counts,
        qpos_axes=weights.qpos_joint_axes,
        qpos_anchors=weights.qpos_joint_anchors,
        hinge_mask=weights.hinge_mask,
        slide_mask=weights.slide_mask,
        link_joint_indices=weights.link_joint_indices,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        xp=np,
    )


def forward_meshes(
    weights: MyoFullBodyWeights,
    body_pose: Float[np.ndarray, "B Q"],
    global_translation: Float[np.ndarray, "B 3"] | None = None,
    *,
    global_rotation: Float[np.ndarray, "B 3"] | None = None,
):
    return core.forward_meshes(
        vertices=weights.vertices,
        faces=weights.faces,
        local_offsets=weights.local_offsets,
        rest_local_rotations=weights.rest_local_rotations,
        parents=weights.parents,
        body_qpos_starts=weights.body_qpos_starts,
        body_qpos_counts=weights.body_qpos_counts,
        qpos_axes=weights.qpos_joint_axes,
        qpos_anchors=weights.qpos_joint_anchors,
        hinge_mask=weights.hinge_mask,
        slide_mask=weights.slide_mask,
        link_joint_indices=weights.link_joint_indices,
        link_vertex_starts=weights.link_vertex_starts,
        link_vertex_counts=weights.link_vertex_counts,
        link_face_starts=weights.link_face_starts,
        link_face_counts=weights.link_face_counts,
        link_geom_positions=weights.link_geom_positions,
        link_geom_rotations=weights.link_geom_rotations,
        body_pose=body_pose,
        global_translation=global_translation,
        global_rotation=global_rotation,
        xp=np,
    )


def world_sites(weights: MyoFullBodyWeights, skeleton):
    return core.world_sites(
        skeleton=skeleton,
        site_positions=weights.site_positions,
        site_body_indices=weights.site_body_indices,
        xp=np,
    )
