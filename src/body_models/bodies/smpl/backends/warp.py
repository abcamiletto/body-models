"""Warp-accelerated SMPL Torch backend."""

import contextlib
import io
import itertools

import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor
from torch.compiler import disable as disable_compile

from body_models.rotations import RotationType
from body_models.bodies.smpl.backends import core
from body_models.bodies.smpl.backends import torch as torch_backend
from body_models.bodies.smpl.io import SmplWeights

__all__ = ["forward_vertices", "forward_skeleton", "prepare_identity", "prepare_pose"]


def prepare_identity(
    weights: SmplWeights,
    shape: Float[Tensor, "*batch 10"],
    skip_vertices: bool = False,
) -> core.SmplIdentity:
    return torch_backend.prepare_identity(weights, shape, skip_vertices=skip_vertices)


def prepare_pose(
    weights: SmplWeights,
    body_pose: Float[Tensor, "*batch 23 N"] | Float[Tensor, "*batch 23 3 3"],
    pelvis_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    rotation_type: RotationType = "axis_angle",
    *,
    local_joint_offsets: Float[Tensor, "*batch J 3"],
    rest_joints: Float[Tensor, "*batch J 3"],
    skip_vertices: bool = False,
) -> core.SmplPreparedPose:
    """Precompute pose-dependent state for repeated forward passes."""
    return torch_backend.prepare_pose(
        weights,
        body_pose,
        pelvis_rotation,
        rotation_type=rotation_type,
        local_joint_offsets=local_joint_offsets,
        rest_joints=rest_joints,
        skip_vertices=skip_vertices,
    )


def forward_vertices(
    weights: SmplWeights,
    rest_vertices: Float[Tensor, "*batch V 3"],
    skinning_transforms: Float[Tensor, "*batch J 4 4"],
    pose_offsets: Float[Tensor, "*batch V 3"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    vertex_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    v_shaped = rest_vertices + pose_offsets
    joint_indices = weights.lbs_joint_indices
    joint_weights = weights.lbs_joint_weights
    if vertex_indices is not None:
        v_shaped = v_shaped[..., vertex_indices, :]
        joint_indices = joint_indices[vertex_indices]
        joint_weights = joint_weights[vertex_indices]

    v_posed = warp_affine_blend_skinning(v_shaped, skinning_transforms, joint_indices, joint_weights)
    return core.apply_global_transform(torch, v_posed, global_rotation, global_translation, rotation_type)


def forward_skeleton(
    weights: SmplWeights,
    skeleton_transforms: Float[Tensor, "*batch J 4 4"],
    global_rotation: Float[Tensor, "*batch N"] | Float[Tensor, "*batch 3 3"] | None = None,
    global_translation: Float[Tensor, "*batch 3"] | None = None,
    joint_indices: list[int] | None = None,
    rotation_type: RotationType = "axis_angle",
):
    return core.forward_skeleton(
        parents=weights.parents,
        global_rotation=global_rotation,
        global_translation=global_translation,
        joint_indices=joint_indices,
        rotation_type=rotation_type,
        skeleton_transforms=skeleton_transforms,
        xp=torch,
    )


@disable_compile
def warp_forward_kinematics(
    rotations: Float[Tensor, "*batch J 3 3"],
    translations: Float[Tensor, "*batch J 3"],
    parents: Tensor,
) -> Float[Tensor, "*batch J 4 4"]:
    _init_warp()
    if rotations.ndim > 4:
        output = torch.empty((*rotations.shape[:-2], 4, 4), device=rotations.device, dtype=rotations.dtype)
        for batch_index in itertools.product(*[range(size) for size in rotations.shape[:-3]]):
            batch_rotations = rotations[batch_index][None]
            batch_translations = translations[batch_index][None]
            output[batch_index] = warp_forward_kinematics(batch_rotations, batch_translations, parents)[0]
        return output

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


@disable_compile
def warp_affine_blend_skinning(
    vertices: Float[Tensor, "*batch V 3"],
    transforms: Float[Tensor, "*batch J 4 4"],
    joint_indices: Tensor,
    joint_weights: Tensor,
) -> Float[Tensor, "*batch V 3"]:
    _check_warp_inputs(vertices)
    if transforms.requires_grad:
        raise ValueError("kernel='warp' is an inference-only forward path; use the default Torch kernel for gradients.")
    if vertices.ndim > 3:
        output = torch.empty_like(vertices)
        for batch_index in itertools.product(*[range(size) for size in vertices.shape[:-2]]):
            batch_vertices = vertices[batch_index][None]
            batch_transforms = transforms[batch_index][None]
            output[batch_index] = warp_affine_blend_skinning(
                batch_vertices, batch_transforms, joint_indices, joint_weights
            )[0]
        return output

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
