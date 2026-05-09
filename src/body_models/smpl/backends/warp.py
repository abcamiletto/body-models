"""Warp-accelerated SMPL Torch backend."""

import contextlib
import io

import torch
import warp as wp
from jaxtyping import Float
from nanomanifold import SO3
from torch import Tensor

from body_models import common
from body_models.rotations import RotationType
from body_models.smpl.backends import core
from body_models.smpl.backends import torch as torch_backend
from body_models.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton"]


def forward_vertices(
    weights: SmplWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    if shape.device.type != "cuda":
        return torch_backend.forward_vertices(
            weights=weights,
            shape=shape,
            body_pose=body_pose,
            pelvis_rotation=pelvis_rotation,
            global_rotation=global_rotation,
            global_translation=global_translation,
            vertex_indices=vertex_indices,
            rotation_type=rotation_type,
        )

    v_shaped, j_t, T_world = _forward_unskinned_vertices_with_warp_fk(
        weights=weights,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        vertex_indices=vertex_indices,
        rotation_type=rotation_type,
    )
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    v_posed = warp_linear_blend_skinning(v_shaped, j_t, T_world, joint_indices, joint_weights)
    return core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)


def _forward_unskinned_vertices_with_warp_fk(
    weights: SmplWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None,
    vertex_indices: list[int] | None,
    rotation_type: RotationType,
) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B 24 3"], Float[Tensor, "B 24 4 4"]]:
    batch_size = body_pose.shape[0]
    if shape.shape[0] == 1 and batch_size > 1:
        shape = torch.broadcast_to(shape, (batch_size, shape.shape[1]))

    body_pose_matrices = SO3.convert(body_pose, src=rotation_type, dst="rotmat", xp=torch)
    if pelvis_rotation is None:
        pelvis_matrices = SO3.identity_as(
            body_pose_matrices,
            batch_dims=(batch_size, 1),
            rotation_type="rotmat",
            xp=torch,
        )
    else:
        pelvis_matrices = SO3.convert(pelvis_rotation, src=rotation_type, dst="rotmat", xp=torch)[:, None]
    pose_matrices = torch.concat([pelvis_matrices, body_pose_matrices], dim=1)

    j_t = weights.j_template + torch.einsum("...p,jdp->...jd", shape, weights.j_shapedirs[:, :, : shape.shape[-1]])
    v_template = weights.v_template
    shapedirs = weights.shapedirs
    posedirs = weights.posedirs
    if vertex_indices is not None:
        vertex_indices_tensor = torch.as_tensor(vertex_indices, device=shape.device)
        v_template = v_template[vertex_indices_tensor]
        shapedirs = shapedirs[vertex_indices_tensor]
        posedirs = posedirs.reshape(posedirs.shape[0], -1, 3)[:, vertex_indices_tensor].reshape(posedirs.shape[0], -1)

    v_t = v_template + torch.einsum("bi,vdi->bvd", shape, shapedirs[:, :, : shape.shape[-1]])
    t_local = torch.empty_like(j_t)
    t_local[:, 0] = j_t[:, 0]
    t_local[:, 1:] = j_t[:, 1:] - j_t[:, weights.parents[1:]]
    parents = torch.as_tensor(weights.parents, device=shape.device, dtype=torch.int32)
    T_world = warp_forward_kinematics(pose_matrices, t_local, parents)

    eye3 = common.eye_as(pose_matrices, batch_dims=(batch_size, 1), xp=torch)
    pose_delta = (pose_matrices[:, 1:] - eye3).reshape(batch_size, -1)
    v_shaped = v_t + (pose_delta @ posedirs).reshape(batch_size, -1, 3)
    return v_shaped, j_t, T_world


def forward_skeleton(
    weights: SmplWeights,
    shape: Float[Tensor, "B 10"],
    body_pose: Float[Tensor, "B 23 N"] | Float[Tensor, "B 23 3 3"],
    pelvis_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_rotation: Float[Tensor, "B N"] | Float[Tensor, "B 3 3"] | None = None,
    global_translation: Float[Tensor, "B 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return core.forward_skeleton(
        j_template=weights.j_template,
        j_shapedirs=weights.j_shapedirs,
        parents=weights.parents,
        kinematic_fronts=weights.kinematic_fronts,
        shape=shape,
        body_pose=body_pose,
        pelvis_rotation=pelvis_rotation,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        xp=torch,
    )


def warp_linear_blend_skinning(
    vertices: Float[Tensor, "B V 3"],
    joints: Float[Tensor, "B J 3"],
    transforms: Float[Tensor, "B J 4 4"],
    joint_indices: Tensor,
    joint_weights: Tensor,
) -> Float[Tensor, "B V 3"]:
    _check_warp_inputs(vertices)
    if vertices.requires_grad or transforms.requires_grad or joints.requires_grad:
        raise ValueError("kernel='warp' is an inference-only forward path; use the default Torch kernel for gradients.")

    _init_warp()
    vertices = vertices.contiguous()
    joints = joints.contiguous()
    transforms = transforms.contiguous()
    joint_indices = joint_indices.to(device=vertices.device, dtype=torch.int32).contiguous()
    joint_weights = joint_weights.to(device=vertices.device, dtype=vertices.dtype).contiguous()
    output = torch.empty_like(vertices)

    flat_vertices = vertices.reshape(-1)
    flat_joints = joints.reshape(-1)
    flat_transforms = transforms.reshape(-1)
    flat_indices = joint_indices.reshape(-1)
    flat_weights = joint_weights.reshape(-1)
    flat_output = output.reshape(-1)

    batch_size, num_vertices = vertices.shape[:2]
    num_joints = joints.shape[1]
    num_slots = joint_indices.shape[1]
    device = wp.from_torch(flat_vertices).device
    wp.launch(
        _skin_vertices_kernel,
        dim=(batch_size, num_vertices),
        inputs=[
            wp.from_torch(flat_vertices),
            wp.from_torch(flat_joints),
            wp.from_torch(flat_transforms),
            wp.from_torch(flat_indices),
            wp.from_torch(flat_weights),
            num_vertices,
            num_joints,
            num_slots,
            wp.from_torch(flat_output),
        ],
        device=device,
    )
    return output


def warp_forward_kinematics(
    rotations: Float[Tensor, "B J 3 3"],
    translations: Float[Tensor, "B J 3"],
    parents: Tensor,
) -> Float[Tensor, "B J 4 4"]:
    _init_warp()
    rotations = rotations.contiguous()
    translations = translations.contiguous()
    parents = parents.to(device=rotations.device, dtype=torch.int32).contiguous()
    output = torch.empty((*rotations.shape[:2], 4, 4), device=rotations.device, dtype=rotations.dtype)
    flat_rotations = rotations.reshape(-1)
    flat_translations = translations.reshape(-1)
    flat_output = output.reshape(-1)

    batch_size, num_joints = translations.shape[:2]
    device = wp.from_torch(flat_rotations).device
    wp.launch(
        _forward_kinematics_kernel,
        dim=batch_size,
        inputs=[
            wp.from_torch(flat_rotations),
            wp.from_torch(flat_translations),
            wp.from_torch(parents),
            num_joints,
            wp.from_torch(flat_output),
        ],
        device=device,
    )
    return output


def warp_affine_blend_skinning(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    joint_indices: Tensor,
    joint_weights: Tensor,
) -> Float[Tensor, "B V 3"]:
    _check_warp_inputs(vertices)
    if transforms.requires_grad:
        raise ValueError("kernel='warp' is an inference-only forward path; use the default Torch kernel for gradients.")
    _init_warp()
    vertices = vertices.contiguous()
    transforms = transforms.contiguous()
    joint_indices = joint_indices.to(device=vertices.device, dtype=torch.int32).contiguous()
    joint_weights = joint_weights.to(device=vertices.device, dtype=vertices.dtype).contiguous()
    output = torch.empty_like(vertices)

    flat_vertices = vertices.reshape(-1)
    flat_transforms = transforms.reshape(-1)
    flat_indices = joint_indices.reshape(-1)
    flat_weights = joint_weights.reshape(-1)
    flat_output = output.reshape(-1)

    batch_size, num_vertices = vertices.shape[:2]
    num_joints = transforms.shape[1]
    num_slots = joint_indices.shape[1]
    device = wp.from_torch(flat_vertices).device
    wp.launch(
        _skin_affine_vertices_kernel,
        dim=(batch_size, num_vertices),
        inputs=[
            wp.from_torch(flat_vertices),
            wp.from_torch(flat_transforms),
            wp.from_torch(flat_indices),
            wp.from_torch(flat_weights),
            num_vertices,
            num_joints,
            num_slots,
            wp.from_torch(flat_output),
        ],
        device=device,
    )
    return output


def _check_warp_inputs(vertices: Tensor) -> None:
    if vertices.dtype != torch.float32:
        raise TypeError("kernel='warp' currently supports float32 tensors only.")
    if vertices.requires_grad:
        raise ValueError("kernel='warp' is an inference-only forward path; use the default Torch kernel for gradients.")


def _init_warp() -> None:
    wp.config.quiet = True
    with contextlib.redirect_stdout(io.StringIO()):
        wp.init()


@wp.kernel
def _forward_kinematics_kernel(
    rotations: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    translations: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    parents: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    num_joints: int,
    output: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    batch = wp.tid()
    for joint in range(num_joints):
        parent = parents[joint]
        local_r = (batch * num_joints + joint) * 9  # ty: ignore[unsupported-operator]
        local_t = (batch * num_joints + joint) * 3  # ty: ignore[unsupported-operator]
        out = (batch * num_joints + joint) * 16  # ty: ignore[unsupported-operator]

        if parent < 0:
            output[out] = rotations[local_r]
            output[out + 1] = rotations[local_r + 1]
            output[out + 2] = rotations[local_r + 2]
            output[out + 3] = translations[local_t]
            output[out + 4] = rotations[local_r + 3]
            output[out + 5] = rotations[local_r + 4]
            output[out + 6] = rotations[local_r + 5]
            output[out + 7] = translations[local_t + 1]
            output[out + 8] = rotations[local_r + 6]
            output[out + 9] = rotations[local_r + 7]
            output[out + 10] = rotations[local_r + 8]
            output[out + 11] = translations[local_t + 2]
        else:
            p = (batch * num_joints + parent) * 16
            lr00 = rotations[local_r]
            lr01 = rotations[local_r + 1]
            lr02 = rotations[local_r + 2]
            lr10 = rotations[local_r + 3]
            lr11 = rotations[local_r + 4]
            lr12 = rotations[local_r + 5]
            lr20 = rotations[local_r + 6]
            lr21 = rotations[local_r + 7]
            lr22 = rotations[local_r + 8]

            pr00 = output[p]
            pr01 = output[p + 1]
            pr02 = output[p + 2]
            pr10 = output[p + 4]
            pr11 = output[p + 5]
            pr12 = output[p + 6]
            pr20 = output[p + 8]
            pr21 = output[p + 9]
            pr22 = output[p + 10]
            tx = translations[local_t]
            ty = translations[local_t + 1]
            tz = translations[local_t + 2]

            output[out] = pr00 * lr00 + pr01 * lr10 + pr02 * lr20
            output[out + 1] = pr00 * lr01 + pr01 * lr11 + pr02 * lr21
            output[out + 2] = pr00 * lr02 + pr01 * lr12 + pr02 * lr22
            output[out + 3] = pr00 * tx + pr01 * ty + pr02 * tz + output[p + 3]
            output[out + 4] = pr10 * lr00 + pr11 * lr10 + pr12 * lr20
            output[out + 5] = pr10 * lr01 + pr11 * lr11 + pr12 * lr21
            output[out + 6] = pr10 * lr02 + pr11 * lr12 + pr12 * lr22
            output[out + 7] = pr10 * tx + pr11 * ty + pr12 * tz + output[p + 7]
            output[out + 8] = pr20 * lr00 + pr21 * lr10 + pr22 * lr20
            output[out + 9] = pr20 * lr01 + pr21 * lr11 + pr22 * lr21
            output[out + 10] = pr20 * lr02 + pr21 * lr12 + pr22 * lr22
            output[out + 11] = pr20 * tx + pr21 * ty + pr22 * tz + output[p + 11]

        output[out + 12] = 0.0
        output[out + 13] = 0.0
        output[out + 14] = 0.0
        output[out + 15] = 1.0


@wp.kernel
def _skin_vertices_kernel(
    vertices: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    joints: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    transforms: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    joint_indices: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    joint_weights: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    num_vertices: int,
    num_joints: int,
    num_slots: int,
    output: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    batch, vertex = wp.tid()  # ty: ignore[invalid-assignment, not-iterable]

    vertex_base = (batch * num_vertices + vertex) * 3
    vx = vertices[vertex_base]
    vy = vertices[vertex_base + 1]
    vz = vertices[vertex_base + 2]
    out_x = float(0.0)
    out_y = float(0.0)
    out_z = float(0.0)

    for slot in range(num_slots):
        slot_index = vertex * num_slots + slot
        joint = joint_indices[slot_index]
        if joint < 0:
            continue

        weight = joint_weights[slot_index]
        joint_base = (batch * num_joints + joint) * 3
        jx = joints[joint_base]
        jy = joints[joint_base + 1]
        jz = joints[joint_base + 2]

        transform_base = (batch * num_joints + joint) * 16
        r00 = transforms[transform_base]
        r01 = transforms[transform_base + 1]
        r02 = transforms[transform_base + 2]
        tx = transforms[transform_base + 3]
        r10 = transforms[transform_base + 4]
        r11 = transforms[transform_base + 5]
        r12 = transforms[transform_base + 6]
        ty = transforms[transform_base + 7]
        r20 = transforms[transform_base + 8]
        r21 = transforms[transform_base + 9]
        r22 = transforms[transform_base + 10]
        tz = transforms[transform_base + 11]

        dx = vx - jx
        dy = vy - jy
        dz = vz - jz
        out_x += weight * (r00 * dx + r01 * dy + r02 * dz + tx)
        out_y += weight * (r10 * dx + r11 * dy + r12 * dz + ty)
        out_z += weight * (r20 * dx + r21 * dy + r22 * dz + tz)

    output[vertex_base] = out_x
    output[vertex_base + 1] = out_y
    output[vertex_base + 2] = out_z


@wp.kernel
def _skin_affine_vertices_kernel(
    vertices: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    transforms: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    joint_indices: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    joint_weights: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    num_vertices: int,
    num_joints: int,
    num_slots: int,
    output: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    batch, vertex = wp.tid()  # ty: ignore[invalid-assignment, not-iterable]

    vertex_base = (batch * num_vertices + vertex) * 3
    vx = vertices[vertex_base]
    vy = vertices[vertex_base + 1]
    vz = vertices[vertex_base + 2]
    out_x = float(0.0)
    out_y = float(0.0)
    out_z = float(0.0)

    for slot in range(num_slots):
        slot_index = vertex * num_slots + slot
        joint = joint_indices[slot_index]
        if joint < 0:
            continue

        weight = joint_weights[slot_index]
        transform_base = (batch * num_joints + joint) * 16
        out_x += weight * (
            transforms[transform_base] * vx
            + transforms[transform_base + 1] * vy
            + transforms[transform_base + 2] * vz
            + transforms[transform_base + 3]
        )
        out_y += weight * (
            transforms[transform_base + 4] * vx
            + transforms[transform_base + 5] * vy
            + transforms[transform_base + 6] * vz
            + transforms[transform_base + 7]
        )
        out_z += weight * (
            transforms[transform_base + 8] * vx
            + transforms[transform_base + 9] * vy
            + transforms[transform_base + 10] * vz
            + transforms[transform_base + 11]
        )

    output[vertex_base] = out_x
    output[vertex_base + 1] = out_y
    output[vertex_base + 2] = out_z
