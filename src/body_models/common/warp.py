"""Differentiable Warp kernels shared by skinned models."""

import contextlib
import functools
import io

import torch
import warp as wp
from jaxtyping import Float, Int
from torch import Tensor
from torch.compiler import disable as disable_compile

__all__ = ["compact_linear_blend_skinning"]


def _require_float32(**tensors: Float[Tensor, "..."]) -> None:
    invalid = [name for name, tensor in tensors.items() if tensor.dtype != torch.float32]
    if invalid:
        names = ", ".join(invalid)
        raise TypeError(f"Warp kernels require float32 tensors; got another dtype for {names}.")


@disable_compile
def compact_linear_blend_skinning(
    vertices: Float[Tensor, "*batch V 3"],
    transforms: Float[Tensor, "*batch J 4 4"],
    *,
    joint_indices: Int[Tensor, "V K"],
    joint_weights: Float[Tensor, "V K"],
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


def _transform_gradients(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    grad_output: Float[Tensor, "B V 3"],
    joint_indices: Int[Tensor, "V K"],
    joint_weights: Float[Tensor, "V K"],
) -> Float[Tensor, "B J 4 4"]:
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


def _launch_affine_blend_skinning(
    vertices: Float[Tensor, "B V 3"],
    transforms: Float[Tensor, "B J 4 4"],
    joint_indices: Int[Tensor, "V K"],
    joint_weights: Float[Tensor, "V K"],
    output: Float[Tensor, "B V 3"],
) -> None:
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


def _from_torch(tensor: Float[Tensor, "..."] | Int[Tensor, "..."]):
    return wp.from_torch(tensor, requires_grad=False)


def _torch_stream(tensor: Float[Tensor, "..."]):
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
