"""Differentiable Warp kernels shared by skinned models."""

import contextlib
import functools
import io
import weakref
from dataclasses import dataclass

import torch
import warp as wp
from jaxtyping import Float, Int
from torch import Tensor
from torch.compiler import disable as disable_compile

__all__ = ["compact_linear_blend_skinning"]

_TRANSFORM_GRADIENT_CHUNK_SIZE = 32


@dataclass(frozen=True)
class _TransformGradientPlan:
    permutation: Int[Tensor, "N"]
    chunk_starts: Int[Tensor, "C"]
    chunk_ends: Int[Tensor, "C"]
    chunk_joints: Int[Tensor, "C"]


_TRANSFORM_GRADIENT_PLANS: dict[
    int,
    tuple[weakref.ReferenceType[Tensor], _TransformGradientPlan],
] = {}


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
    output = _WarpCompactLinearBlendSkinning.apply(
        flat_vertices,
        flat_transforms,
        joint_indices,
        joint_weights,
    )
    return output.reshape(*batch_shape, num_vertices, 3)


class _WarpCompactLinearBlendSkinning(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, transforms, joint_indices, joint_weights):
        output = torch.empty_like(vertices)
        _launch_compact_linear_blend_skinning(vertices, transforms, joint_indices, joint_weights, output)
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
                    _compact_linear_blend_skinning_backward_vertices_kernel,
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
    plan = _transform_gradient_plan(joint_indices)
    grad_transforms = torch.zeros_like(transforms)
    if plan.permutation.numel() == 0:
        return grad_transforms

    with _torch_stream(vertices):
        wp.launch(
            _compact_linear_blend_skinning_backward_transforms_kernel,
            dim=(vertices.shape[0], plan.chunk_starts.shape[0], 12),
            inputs=[
                _from_torch(vertices.reshape(-1)),
                _from_torch(grad_output.reshape(-1)),
                _from_torch(joint_weights.reshape(-1)),
                _from_torch(plan.permutation),
                _from_torch(plan.chunk_starts),
                _from_torch(plan.chunk_ends),
                _from_torch(plan.chunk_joints),
                vertices.shape[1],
                transforms.shape[1],
                joint_indices.shape[1],
                _from_torch(grad_transforms.reshape(-1)),
            ],
            device=_from_torch(vertices).device,
        )
    return grad_transforms


def _transform_gradient_plan(joint_indices: Int[Tensor, "V K"]) -> _TransformGradientPlan:
    key = id(joint_indices)
    cached = _TRANSFORM_GRADIENT_PLANS.get(key)
    if cached is not None and cached[0]() is joint_indices:
        return cached[1]

    plan = _build_transform_gradient_plan(joint_indices)

    def remove(reference: weakref.ReferenceType[Tensor]) -> None:
        current = _TRANSFORM_GRADIENT_PLANS.get(key)
        if current is not None and current[0] is reference:
            del _TRANSFORM_GRADIENT_PLANS[key]

    reference = weakref.ref(joint_indices, remove)
    _TRANSFORM_GRADIENT_PLANS[key] = reference, plan
    return plan


def _build_transform_gradient_plan(joint_indices: Int[Tensor, "V K"]) -> _TransformGradientPlan:
    flat_indices = joint_indices.reshape(-1)
    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).flatten()
    permutation = valid_positions[torch.argsort(flat_indices[valid_positions])]
    sorted_joints = flat_indices[permutation]
    joints, counts = torch.unique_consecutive(sorted_joints, return_counts=True)

    chunks_per_joint = torch.div(
        counts + _TRANSFORM_GRADIENT_CHUNK_SIZE - 1,
        _TRANSFORM_GRADIENT_CHUNK_SIZE,
        rounding_mode="floor",
    )
    chunk_joints = torch.repeat_interleave(joints, chunks_per_joint)
    joint_starts = torch.cumsum(counts, dim=0) - counts
    chunk_group_starts = torch.cumsum(chunks_per_joint, dim=0) - chunks_per_joint
    chunk_offsets = torch.arange(chunk_joints.shape[0], device=joint_indices.device)
    chunk_offsets -= torch.repeat_interleave(chunk_group_starts, chunks_per_joint)
    chunk_starts = torch.repeat_interleave(joint_starts, chunks_per_joint)
    chunk_starts += chunk_offsets * _TRANSFORM_GRADIENT_CHUNK_SIZE
    joint_ends = torch.repeat_interleave(joint_starts + counts, chunks_per_joint)
    chunk_ends = torch.minimum(
        chunk_starts + _TRANSFORM_GRADIENT_CHUNK_SIZE,
        joint_ends,
    )
    return _TransformGradientPlan(
        permutation=permutation.to(torch.int32).contiguous(),
        chunk_starts=chunk_starts.to(torch.int32).contiguous(),
        chunk_ends=chunk_ends.to(torch.int32).contiguous(),
        chunk_joints=chunk_joints.to(torch.int32).contiguous(),
    )


def _launch_compact_linear_blend_skinning(
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
            _compact_linear_blend_skinning_kernel,
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
def _compact_linear_blend_skinning_kernel(
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
def _compact_linear_blend_skinning_backward_vertices_kernel(
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


@wp.kernel
def _compact_linear_blend_skinning_backward_transforms_kernel(
    vertices: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    grad_output: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    joint_weights: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
    permutation: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    chunk_starts: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    chunk_ends: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    chunk_joints: wp.array(dtype=wp.int32),  # ty: ignore[invalid-type-form]
    num_vertices: int,
    num_joints: int,
    num_slots: int,
    grad_transforms: wp.array(dtype=wp.float32),  # ty: ignore[invalid-type-form]
):
    batch, chunk, component = wp.tid()  # ty: ignore[invalid-assignment, not-iterable]
    row = component // 4
    column = component - row * 4
    total = float(0.0)

    for influence in range(chunk_starts[chunk], chunk_ends[chunk]):
        flat_index = permutation[influence]
        vertex = flat_index // num_slots
        vertex_base = (batch * num_vertices + vertex) * 3
        coordinate = float(1.0)
        if column < 3:
            coordinate = vertices[vertex_base + column]
        total += joint_weights[flat_index] * grad_output[vertex_base + row] * coordinate

    joint = chunk_joints[chunk]
    transform_base = (batch * num_joints + joint) * 16
    wp.atomic_add(grad_transforms, transform_base + component, total)
