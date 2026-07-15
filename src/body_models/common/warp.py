"""Differentiable Warp kernels shared by skinned models."""

import contextlib
import functools
import io

import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor
from torch.compiler import disable as disable_compile

__all__ = ["compact_linear_blend_skinning", "forward_kinematics"]


def _require_float32(**tensors: Tensor) -> None:
    invalid = [name for name, tensor in tensors.items() if tensor.dtype != torch.float32]
    if invalid:
        names = ", ".join(invalid)
        raise TypeError(f"Warp kernels require float32 tensors; got another dtype for {names}.")


@disable_compile
def forward_kinematics(
    rotations: Float[Tensor, "*batch J 3 3"],
    translations: Float[Tensor, "*batch J 3"],
    parents: Tensor,
) -> Float[Tensor, "*batch J 4 4"]:
    """Compose float32 Torch joint transforms with Warp."""
    _require_float32(rotations=rotations, translations=translations)
    _init_warp()
    batch_shape = rotations.shape[:-3]
    num_joints = rotations.shape[-3]
    rotations = rotations.reshape(-1, num_joints, 3, 3)
    translations = translations.reshape(-1, num_joints, 3)
    parents = parents.to(device=rotations.device, dtype=torch.int32).contiguous()
    output = _WarpForwardKinematics.apply(rotations, translations, parents)
    return output.reshape(*batch_shape, num_joints, 4, 4)


class _WarpForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotations, translations, parents):
        rotations = rotations.contiguous()
        translations = translations.contiguous()
        output = torch.empty((*rotations.shape[:2], 4, 4), device=rotations.device, dtype=rotations.dtype)

        wp_rotations = wp.from_torch(rotations.reshape(-1), requires_grad=ctx.needs_input_grad[0])
        wp_translations = wp.from_torch(translations.reshape(-1), requires_grad=ctx.needs_input_grad[1])
        wp_parents = wp.from_torch(parents)
        needs_backward = any(ctx.needs_input_grad[:2])
        wp_output = wp.from_torch(output.reshape(-1), requires_grad=needs_backward)

        batch_size, num_joints = translations.shape[:2]
        tape = wp.Tape() if needs_backward else None
        with tape if tape is not None else contextlib.nullcontext():
            wp.launch(
                _forward_kinematics_kernel,
                dim=batch_size,
                inputs=[wp_rotations, wp_translations, wp_parents, num_joints, wp_output],
                device=wp_rotations.device,
            )

        if needs_backward:
            ctx.tape = tape
            ctx.wp_inputs = wp_rotations, wp_translations
            ctx.wp_output = wp_output
            ctx.input_shapes = rotations.shape, translations.shape
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        wp_grad_output = wp.from_torch(grad_output.contiguous().reshape(-1))
        ctx.tape.backward(grads={ctx.wp_output: wp_grad_output})
        wp_rotations, wp_translations = ctx.wp_inputs
        rotation_shape, translation_shape = ctx.input_shapes
        grad_rotations = wp.to_torch(wp_rotations.grad).reshape(rotation_shape) if wp_rotations.requires_grad else None
        grad_translations = (
            wp.to_torch(wp_translations.grad).reshape(translation_shape) if wp_translations.requires_grad else None
        )
        return grad_rotations, grad_translations, None


@disable_compile
def compact_linear_blend_skinning(
    vertices: Float[Tensor, "*batch V 3"],
    transforms: Float[Tensor, "*batch J 4 4"],
    *,
    joint_indices: Tensor,
    joint_weights: Tensor,
) -> Float[Tensor, "*batch V 3"]:
    """Apply sparse float32 linear blend skinning with Warp autograd."""
    _require_float32(vertices=vertices, transforms=transforms)
    _init_warp()
    batch_shape = torch.broadcast_shapes(vertices.shape[:-2], transforms.shape[:-3])
    vertices = vertices.expand(*batch_shape, *vertices.shape[-2:])
    transforms = transforms.expand(*batch_shape, *transforms.shape[-3:])
    vertices = vertices.contiguous()
    transforms = transforms.contiguous()
    joint_indices = joint_indices.to(device=vertices.device, dtype=torch.int32).contiguous()
    joint_weights = joint_weights.to(device=vertices.device, dtype=vertices.dtype).contiguous()
    num_vertices = vertices.shape[-2]
    num_joints = transforms.shape[-3]
    flat_vertices = vertices.reshape(-1, num_vertices, 3)
    flat_transforms = transforms.reshape(-1, num_joints, 4, 4)
    output = _WarpAffineBlendSkinning.apply(
        flat_vertices,
        flat_transforms,
        joint_indices,
        joint_weights,
    )
    return output.reshape(*batch_shape, num_vertices, 3)


class _WarpAffineBlendSkinning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, transforms, joint_indices, joint_weights):
        output = torch.empty_like(vertices)
        _launch_affine_blend_skinning(vertices, transforms, joint_indices, joint_weights, output)
        ctx.save_for_backward(vertices, transforms, joint_indices, joint_weights)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        vertices, transforms, joint_indices, joint_weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_vertices = grad_transforms = None

        if ctx.needs_input_grad[0]:
            grad_vertices = torch.empty_like(vertices)
            with _torch_stream(vertices):
                wp.launch(
                    _skin_affine_vertices_backward_vertices_kernel,
                    dim=vertices.shape[:2],
                    inputs=[
                        _from_torch(grad_output.reshape(-1)),
                        _from_torch(transforms.reshape(-1)),
                        _from_torch(joint_indices.reshape(-1)),
                        _from_torch(joint_weights.reshape(-1)),
                        vertices.shape[1],
                        transforms.shape[1],
                        joint_indices.shape[1],
                        _from_torch(grad_vertices.reshape(-1)),
                    ],
                    device=_from_torch(vertices).device,
                )

        if ctx.needs_input_grad[1]:
            grad_transforms = _transform_gradients(
                vertices,
                transforms,
                grad_output,
                joint_indices,
                joint_weights,
            )

        return grad_vertices, grad_transforms, None, None


def _transform_gradients(vertices, transforms, grad_output, joint_indices, joint_weights):
    homogeneous_vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    contributions = grad_output[..., None] * homogeneous_vertices[..., None, :]
    grad_transforms = torch.zeros_like(transforms)
    grad_affine = grad_transforms[:, :, :3, :]
    batch_size, num_vertices = vertices.shape[:2]
    for slot in range(joint_indices.shape[1]):
        indices = joint_indices[:, slot]
        valid = indices >= 0
        indices = indices.clamp_min(0).view(1, num_vertices, 1, 1)
        indices = indices.expand(batch_size, num_vertices, 3, 4)
        weights = (joint_weights[:, slot] * valid).view(1, num_vertices, 1, 1)
        grad_affine.scatter_add_(1, indices, contributions * weights)
    return grad_transforms


def _launch_affine_blend_skinning(vertices, transforms, joint_indices, joint_weights, output):
    flat_vertices = vertices.reshape(-1)
    flat_transforms = transforms.reshape(-1)
    flat_indices = joint_indices.reshape(-1)
    flat_weights = joint_weights.reshape(-1)
    flat_output = output.reshape(-1)

    batch_size, num_vertices = vertices.shape[:2]
    num_joints = transforms.shape[1]
    num_slots = joint_indices.shape[1]
    device = _from_torch(flat_vertices).device
    with _torch_stream(vertices):
        wp.launch(
            _skin_affine_vertices_kernel,
            dim=(batch_size, num_vertices),
            inputs=[
                _from_torch(flat_vertices),
                _from_torch(flat_transforms),
                _from_torch(flat_indices),
                _from_torch(flat_weights),
                num_vertices,
                num_joints,
                num_slots,
                _from_torch(flat_output),
            ],
            device=device,
        )


def _from_torch(tensor: Tensor):
    return wp.from_torch(tensor, requires_grad=False)


def _torch_stream(tensor: Tensor):
    if tensor.device.type == "cuda":
        stream = wp.stream_from_torch(torch.cuda.current_stream(tensor.device))
        return wp.ScopedStream(stream)
    return contextlib.nullcontext()


@functools.cache
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


@wp.kernel
def _skin_affine_vertices_backward_vertices_kernel(
    grad_output: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    transforms: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    joint_indices: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    joint_weights: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    num_vertices: int,
    num_joints: int,
    num_slots: int,
    grad_vertices: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    batch, vertex = wp.tid()  # ty: ignore[invalid-assignment, not-iterable]
    vertex_base = (batch * num_vertices + vertex) * 3
    gx = grad_output[vertex_base]
    gy = grad_output[vertex_base + 1]
    gz = grad_output[vertex_base + 2]
    grad_x = float(0.0)
    grad_y = float(0.0)
    grad_z = float(0.0)

    for slot in range(num_slots):
        slot_index = vertex * num_slots + slot
        joint = joint_indices[slot_index]
        if joint < 0:
            continue

        weight = joint_weights[slot_index]
        transform_base = (batch * num_joints + joint) * 16
        grad_x += weight * (
            transforms[transform_base] * gx + transforms[transform_base + 4] * gy + transforms[transform_base + 8] * gz
        )
        grad_y += weight * (
            transforms[transform_base + 1] * gx
            + transforms[transform_base + 5] * gy
            + transforms[transform_base + 9] * gz
        )
        grad_z += weight * (
            transforms[transform_base + 2] * gx
            + transforms[transform_base + 6] * gy
            + transforms[transform_base + 10] * gz
        )

    grad_vertices[vertex_base] = grad_x
    grad_vertices[vertex_base + 1] = grad_y
    grad_vertices[vertex_base + 2] = grad_z
