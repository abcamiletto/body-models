"""SciPy-optimized SOMA kernels for the NumPy backend."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from nanomanifold import SO3
from scipy import sparse

from . import base


def forward_vertices(
    mean_full,
    mean_active,
    shapedirs_full,
    shapedirs_active,
    eigenvalues,
    bind_shape_full,
    skin_weights_active,
    bind_pose_world,
    bind_pose_local,
    t_pose_world,
    joint_regressor,
    joint_children_full,
    joint_children_indices_full,
    skinned_vertex_indices_full,
    skinned_vertex_indices_full_index,
    kinematic_fronts_full,
    parents_full,
    parents_full_index,
    identity,
    pose,
    corrective_bindpose,
    corrective_W1,
    corrective_W2_rows,
    corrective_W2_cols,
    corrective_W2_values,
    rest_shape_full=None,
    rest_shape_active=None,
    world_bind_pose_fit=None,
    global_rotation=None,
    global_translation=None,
    vertex_indices=None,
    vertex_map=None,
    corrective_use_tanh=True,
    apply_correctives=True,
    rotation_type="axis_angle",
    match_warp=True,
    *,
    xp: Any = None,
):
    if xp is None:
        xp = np

    pose_rot = SO3.convert(pose, src=rotation_type, dst="rotmat", xp=xp)
    batch_size = pose_rot.shape[0]
    pose_rot_full = base._orient_pose_rot_full(xp, pose_rot, t_pose_world, parents_full_index)
    rest_shape_full, rest_shape_active, world_bind_pose_fit = base._prepare_rest_shapes(
        xp=xp,
        batch_size=batch_size,
        mean_full=mean_full,
        mean_active=mean_active,
        shapedirs_full=shapedirs_full,
        shapedirs_active=shapedirs_active,
        eigenvalues=eigenvalues,
        bind_shape=bind_shape_full,
        bind_pose_world=bind_pose_world,
        joint_regressor=joint_regressor,
        joint_children_full=joint_children_full,
        joint_children_indices_full=joint_children_indices_full,
        skinned_vertex_indices_full=skinned_vertex_indices_full,
        skinned_vertex_indices_full_index=skinned_vertex_indices_full_index,
        parents_full=parents_full,
        identity=identity,
        rest_shape_full=rest_shape_full,
        rest_shape_active=rest_shape_active,
        world_bind_pose_fit=world_bind_pose_fit,
        match_warp=match_warp,
    )

    skin_weights_sparse = sparse.csr_matrix(skin_weights_active)
    if apply_correctives:
        rest_shape_active, world_bind_pose = repose_to_bind_pose(
            xp=xp,
            rest_shape=rest_shape_active,
            skin_weights=skin_weights_sparse,
            world_bind_pose_fit=world_bind_pose_fit,
            bind_pose_local=bind_pose_local,
            kinematic_fronts=kinematic_fronts_full,
            parents_full_index=parents_full_index,
        )
        corrective_offsets = apply_pose_correctives(
            pose_rot_full=pose_rot_full,
            bindpose=corrective_bindpose,
            W1=corrective_W1,
            W2_rows=corrective_W2_rows,
            W2_cols=corrective_W2_cols,
            W2_values=corrective_W2_values,
            num_vertices=mean_full.shape[0],
            use_tanh=corrective_use_tanh,
            xp=xp,
        )
        if vertex_map is not None:
            corrective_offsets = corrective_offsets[:, vertex_map]
        rest_shape_active = rest_shape_active + corrective_offsets
    else:
        world_bind_pose = world_bind_pose_fit

    verts_cm, _ = pose_mesh_from_oriented_pose(
        xp=xp,
        rest_shape=rest_shape_active,
        skin_weights=skin_weights_sparse,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=kinematic_fronts_full,
        parents_full_index=parents_full_index,
        pose_rot_full=pose_rot_full,
    )

    verts = verts_cm * 0.01
    verts = base._apply_global_transform_vertices(xp, verts, global_rotation, global_translation, rotation_type)
    if vertex_indices is not None:
        verts = verts[:, xp.asarray(vertex_indices)]
    return verts


def apply_pose_correctives(
    pose_rot_full,
    bindpose,
    W1,
    W2_rows,
    W2_cols,
    W2_values,
    num_vertices: int,
    use_tanh: bool,
    *,
    xp: Any = None,
):
    if xp is None:
        xp = np

    batch_size = pose_rot_full.shape[0]
    x = bindpose.swapaxes(-2, -1)[None] @ pose_rot_full
    x = np.asarray(x, copy=True)
    x[:, :, 0, 0] -= 1.0
    x[:, :, 1, 1] -= 1.0
    feat = x[:, :, :, :2].reshape(batch_size, -1)

    z = feat @ W1
    z = xp.maximum(z, xp.asarray(0.0, dtype=feat.dtype))
    if use_tanh:
        z = xp.tanh(z)

    W2 = sparse.csr_matrix((W2_values, (W2_rows, W2_cols)), shape=(W1.shape[1], num_vertices * 3))
    return xp.asarray(z @ W2).reshape(batch_size, num_vertices, 3)


def repose_to_bind_pose(
    xp,
    rest_shape,
    skin_weights,
    world_bind_pose_fit,
    bind_pose_local,
    kinematic_fronts,
    parents_full_index,
):
    T_world = base._repose_skeleton_to_bind_pose(
        xp=xp,
        world_bind_pose_fit=world_bind_pose_fit,
        bind_pose_local=bind_pose_local,
        kinematic_fronts=kinematic_fronts,
        parents_full_index=parents_full_index,
    )
    bone = T_world @ base._invert_transforms(xp, world_bind_pose_fit)
    verts = linear_blend_skinning(xp, rest_shape, skin_weights, bone)
    return verts, T_world


def pose_mesh_from_oriented_pose(
    xp,
    rest_shape,
    skin_weights,
    world_bind_pose,
    kinematic_fronts,
    parents_full_index,
    pose_rot_full,
):
    T_world = base._pose_skeleton_from_oriented_pose(
        xp=xp,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=kinematic_fronts,
        parents_full_index=parents_full_index,
        pose_rot_full=pose_rot_full,
    )
    bone = T_world @ base._invert_transforms(xp, world_bind_pose)
    verts = linear_blend_skinning(xp, rest_shape, skin_weights, bone)
    return verts, T_world


def linear_blend_skinning(xp, bind_shape, skin_weights, bone_transforms):
    R = bone_transforms[..., :3, :3]
    t = bone_transforms[..., :3, 3]
    out = xp.empty_like(bind_shape)
    for batch_index in range(bind_shape.shape[0]):
        R_blend = xp.asarray(skin_weights @ R[batch_index].reshape(R.shape[1], 9)).reshape(-1, 3, 3)
        t_blend = xp.asarray(skin_weights @ t[batch_index])
        out[batch_index] = xp.einsum("vik,vk->vi", R_blend, bind_shape[batch_index]) + t_blend
    return out


ops = replace(
    base.ops,
    forward_vertices=forward_vertices,
    apply_pose_correctives=apply_pose_correctives,
    linear_blend_skinning=linear_blend_skinning,
)
