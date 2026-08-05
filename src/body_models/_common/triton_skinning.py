"""Compiled CUDA LBS with deterministic joint-major gradient reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int
from torch import Tensor, nn

from body_models._common.skinning import CompactSkinningState

__all__ = ["TritonSkinningState", "compact_linear_blend_skinning", "prepare_compact_skinning"]

_BLOCK_SIZE = 256
_TRANSFORM_NUM_WARPS = 8


@dataclass(frozen=True)
class _JointMajorCsrPlan:
    vertex_indices: Int[Tensor, "N"]
    weights: Float[Tensor, "N"]
    offsets: Int[Tensor, "J+1"]


class TritonSkinningState(nn.Module):
    """Compact weights and their joint-major CUDA reduction plan."""

    joint_indices: Int[Tensor, "V K"]
    joint_weights: Float[Tensor, "V K"]
    _plan_vertex_indices: Int[Tensor, "N"]
    _plan_weights: Float[Tensor, "N"]
    _plan_offsets: Int[Tensor, "J+1"]

    def __init__(
        self,
        joint_indices: Tensor,
        joint_weights: Tensor,
        plan: _JointMajorCsrPlan,
    ) -> None:
        super().__init__()
        self.register_buffer("joint_indices", joint_indices, persistent=True)
        self.register_buffer("joint_weights", joint_weights, persistent=True)
        self.register_buffer("_plan_vertex_indices", plan.vertex_indices, persistent=False)
        self.register_buffer("_plan_weights", plan.weights, persistent=False)
        self.register_buffer("_plan_offsets", plan.offsets, persistent=False)
        self.register_load_state_dict_post_hook(_rebuild_transform_gradient_plan)


def prepare_compact_skinning(skinning: CompactSkinningState) -> TritonSkinningState:
    """Materialize compact weights and a joint-major gradient plan."""
    if isinstance(skinning, TritonSkinningState):
        return skinning
    joint_indices = torch.as_tensor(skinning.joint_indices, dtype=torch.int32).clone().contiguous()
    joint_weights = torch.as_tensor(skinning.joint_weights, dtype=torch.float32).clone().contiguous()
    plan = _build_transform_gradient_plan(joint_indices, joint_weights)
    return TritonSkinningState(joint_indices, joint_weights, plan)


def compact_linear_blend_skinning(
    vertices: Float[Tensor, "*batch V 3"],
    transforms: Float[Tensor, "*batch J 4 4"],
    *,
    skinning: TritonSkinningState,
) -> Float[Tensor, "*batch V 3"]:
    """Apply compact float32 LBS with a compiled first-order CUDA backward."""
    state_tensors = (
        skinning.joint_indices,
        skinning.joint_weights,
        skinning._plan_vertex_indices,
        skinning._plan_weights,
        skinning._plan_offsets,
    )
    tensors = (vertices, transforms, *state_tensors)
    if any(not tensor.is_cuda for tensor in tensors):
        raise TypeError("Triton skinning requires all tensors on CUDA")
    if any(tensor.device != vertices.device for tensor in tensors[1:]):
        raise TypeError("Triton skinning requires all tensors on the same CUDA device")

    float_tensors = (vertices, transforms, skinning.joint_weights, skinning._plan_weights)
    if any(tensor.dtype != torch.float32 for tensor in float_tensors):
        raise TypeError("Triton skinning requires float32 tensors")

    batch_shape = torch.broadcast_shapes(vertices.shape[:-2], transforms.shape[:-3])
    vertices = vertices.expand(*batch_shape, *vertices.shape[-2:]).contiguous()
    transforms = transforms.expand(*batch_shape, *transforms.shape[-3:]).contiguous()
    num_vertices = vertices.shape[-2]
    num_joints = transforms.shape[-3]
    flat_vertices = vertices.reshape(-1, num_vertices, 3)
    flat_transforms = transforms.reshape(-1, num_joints, 4, 4)
    output = _compact_lbs_op(
        flat_vertices,
        flat_transforms,
        skinning.joint_indices,
        skinning.joint_weights,
        skinning._plan_vertex_indices,
        skinning._plan_weights,
        skinning._plan_offsets,
    )
    return output.reshape(*batch_shape, num_vertices, 3)


@triton.jit
def _forward_kernel(
    vertices,
    transforms,
    joint_indices,
    joint_weights,
    output,
    total,
    num_vertices: tl.constexpr,
    num_joints: tl.constexpr,
    num_slots: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    batches = offsets // num_vertices
    vertex_indices = offsets - batches * num_vertices
    x = tl.load(vertices + offsets * 3, mask=mask, other=0.0)
    y = tl.load(vertices + offsets * 3 + 1, mask=mask, other=0.0)
    z = tl.load(vertices + offsets * 3 + 2, mask=mask, other=0.0)
    output_x = tl.zeros((block_size,), dtype=tl.float32)
    output_y = tl.zeros((block_size,), dtype=tl.float32)
    output_z = tl.zeros((block_size,), dtype=tl.float32)

    for slot in tl.static_range(0, num_slots):
        joints = tl.load(joint_indices + vertex_indices * num_slots + slot, mask=mask, other=-1)
        valid = joints >= 0
        weights = tl.load(joint_weights + vertex_indices * num_slots + slot, mask=mask & valid, other=0.0)
        transform_base = (batches * num_joints + tl.maximum(joints, 0)) * 16
        r00 = tl.load(transforms + transform_base, mask=mask & valid, other=0.0)
        r01 = tl.load(transforms + transform_base + 1, mask=mask & valid, other=0.0)
        r02 = tl.load(transforms + transform_base + 2, mask=mask & valid, other=0.0)
        tx = tl.load(transforms + transform_base + 3, mask=mask & valid, other=0.0)
        r10 = tl.load(transforms + transform_base + 4, mask=mask & valid, other=0.0)
        r11 = tl.load(transforms + transform_base + 5, mask=mask & valid, other=0.0)
        r12 = tl.load(transforms + transform_base + 6, mask=mask & valid, other=0.0)
        ty = tl.load(transforms + transform_base + 7, mask=mask & valid, other=0.0)
        r20 = tl.load(transforms + transform_base + 8, mask=mask & valid, other=0.0)
        r21 = tl.load(transforms + transform_base + 9, mask=mask & valid, other=0.0)
        r22 = tl.load(transforms + transform_base + 10, mask=mask & valid, other=0.0)
        tz = tl.load(transforms + transform_base + 11, mask=mask & valid, other=0.0)
        output_x += weights * (r00 * x + r01 * y + r02 * z + tx)
        output_y += weights * (r10 * x + r11 * y + r12 * z + ty)
        output_z += weights * (r20 * x + r21 * y + r22 * z + tz)

    tl.store(output + offsets * 3, output_x, mask=mask)
    tl.store(output + offsets * 3 + 1, output_y, mask=mask)
    tl.store(output + offsets * 3 + 2, output_z, mask=mask)


@triton.jit
def _grad_vertices_kernel(
    grad_output,
    transforms,
    joint_indices,
    joint_weights,
    grad_vertices,
    total,
    num_vertices: tl.constexpr,
    num_joints: tl.constexpr,
    num_slots: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    batches = offsets // num_vertices
    vertex_indices = offsets - batches * num_vertices
    grad_x = tl.load(grad_output + offsets * 3, mask=mask, other=0.0)
    grad_y = tl.load(grad_output + offsets * 3 + 1, mask=mask, other=0.0)
    grad_z = tl.load(grad_output + offsets * 3 + 2, mask=mask, other=0.0)
    output_x = tl.zeros((block_size,), dtype=tl.float32)
    output_y = tl.zeros((block_size,), dtype=tl.float32)
    output_z = tl.zeros((block_size,), dtype=tl.float32)

    for slot in tl.static_range(0, num_slots):
        joints = tl.load(joint_indices + vertex_indices * num_slots + slot, mask=mask, other=-1)
        valid = joints >= 0
        weights = tl.load(joint_weights + vertex_indices * num_slots + slot, mask=mask & valid, other=0.0)
        transform_base = (batches * num_joints + tl.maximum(joints, 0)) * 16
        r00 = tl.load(transforms + transform_base, mask=mask & valid, other=0.0)
        r01 = tl.load(transforms + transform_base + 1, mask=mask & valid, other=0.0)
        r02 = tl.load(transforms + transform_base + 2, mask=mask & valid, other=0.0)
        r10 = tl.load(transforms + transform_base + 4, mask=mask & valid, other=0.0)
        r11 = tl.load(transforms + transform_base + 5, mask=mask & valid, other=0.0)
        r12 = tl.load(transforms + transform_base + 6, mask=mask & valid, other=0.0)
        r20 = tl.load(transforms + transform_base + 8, mask=mask & valid, other=0.0)
        r21 = tl.load(transforms + transform_base + 9, mask=mask & valid, other=0.0)
        r22 = tl.load(transforms + transform_base + 10, mask=mask & valid, other=0.0)
        output_x += weights * (r00 * grad_x + r10 * grad_y + r20 * grad_z)
        output_y += weights * (r01 * grad_x + r11 * grad_y + r21 * grad_z)
        output_z += weights * (r02 * grad_x + r12 * grad_y + r22 * grad_z)

    tl.store(grad_vertices + offsets * 3, output_x, mask=mask)
    tl.store(grad_vertices + offsets * 3 + 1, output_y, mask=mask)
    tl.store(grad_vertices + offsets * 3 + 2, output_z, mask=mask)


@triton.jit
def _grad_transforms_kernel(
    grad_output,
    vertices,
    plan_vertex_indices,
    plan_weights,
    plan_offsets,
    grad_transforms,
    num_vertices: tl.constexpr,
    num_joints: tl.constexpr,
    plan_joints: tl.constexpr,
    block_size: tl.constexpr,
):
    program = tl.program_id(0)
    batch = program // num_joints
    joint = program - batch * num_joints
    lanes = tl.arange(0, block_size)
    first = tl.load(plan_offsets + joint, mask=joint < plan_joints, other=0)
    end = tl.load(plan_offsets + joint + 1, mask=joint < plan_joints, other=0)
    grad_r00 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r01 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r02 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r10 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r11 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r12 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r20 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r21 = tl.zeros((block_size,), dtype=tl.float32)
    grad_r22 = tl.zeros((block_size,), dtype=tl.float32)
    grad_tx = tl.zeros((block_size,), dtype=tl.float32)
    grad_ty = tl.zeros((block_size,), dtype=tl.float32)
    grad_tz = tl.zeros((block_size,), dtype=tl.float32)

    for start in tl.range(first, end, block_size, loop_unroll_factor=1):
        influence_offsets = start + lanes
        mask = influence_offsets < end
        vertex_indices = tl.load(plan_vertex_indices + influence_offsets, mask=mask, other=0)
        weights = tl.load(plan_weights + influence_offsets, mask=mask, other=0.0)
        vertex_offsets = batch * num_vertices + vertex_indices
        gx = weights * tl.load(grad_output + vertex_offsets * 3, mask=mask, other=0.0)
        gy = weights * tl.load(grad_output + vertex_offsets * 3 + 1, mask=mask, other=0.0)
        gz = weights * tl.load(grad_output + vertex_offsets * 3 + 2, mask=mask, other=0.0)
        x = tl.load(vertices + vertex_offsets * 3, mask=mask, other=0.0)
        y = tl.load(vertices + vertex_offsets * 3 + 1, mask=mask, other=0.0)
        z = tl.load(vertices + vertex_offsets * 3 + 2, mask=mask, other=0.0)
        grad_r00 += gx * x
        grad_r01 += gx * y
        grad_r02 += gx * z
        grad_r10 += gy * x
        grad_r11 += gy * y
        grad_r12 += gy * z
        grad_r20 += gz * x
        grad_r21 += gz * y
        grad_r22 += gz * z
        grad_tx += gx
        grad_ty += gy
        grad_tz += gz

    output = program * 16
    tl.store(grad_transforms + output, tl.sum(grad_r00))
    tl.store(grad_transforms + output + 1, tl.sum(grad_r01))
    tl.store(grad_transforms + output + 2, tl.sum(grad_r02))
    tl.store(grad_transforms + output + 3, tl.sum(grad_tx))
    tl.store(grad_transforms + output + 4, tl.sum(grad_r10))
    tl.store(grad_transforms + output + 5, tl.sum(grad_r11))
    tl.store(grad_transforms + output + 6, tl.sum(grad_r12))
    tl.store(grad_transforms + output + 7, tl.sum(grad_ty))
    tl.store(grad_transforms + output + 8, tl.sum(grad_r20))
    tl.store(grad_transforms + output + 9, tl.sum(grad_r21))
    tl.store(grad_transforms + output + 10, tl.sum(grad_r22))
    tl.store(grad_transforms + output + 11, tl.sum(grad_tz))


@torch.library.triton_op("body_models::compact_linear_blend_skinning", mutates_args={})
def _compact_lbs_op(
    vertices: Tensor,
    transforms: Tensor,
    joint_indices: Tensor,
    joint_weights: Tensor,
    plan_vertex_indices: Tensor,
    plan_weights: Tensor,
    plan_offsets: Tensor,
) -> Tensor:
    output = torch.empty_like(vertices)
    total = vertices.shape[0] * vertices.shape[1]
    kernel = cast(Any, torch.library.wrap_triton(_forward_kernel))
    kernel[(triton.cdiv(total, _BLOCK_SIZE),)](
        vertices,
        transforms,
        joint_indices,
        joint_weights,
        output,
        total,
        num_vertices=vertices.shape[1],
        num_joints=transforms.shape[1],
        num_slots=joint_indices.shape[1],
        block_size=_BLOCK_SIZE,
    )
    return output


@torch.library.triton_op("body_models::compact_linear_blend_skinning_grad_vertices", mutates_args={})
def _compact_lbs_grad_vertices_op(
    grad_output: Tensor,
    transforms: Tensor,
    joint_indices: Tensor,
    joint_weights: Tensor,
) -> Tensor:
    grad_vertices = torch.empty((*grad_output.shape[:-1], 3), dtype=grad_output.dtype, device=grad_output.device)
    total = grad_output.shape[0] * grad_output.shape[1]
    kernel = cast(Any, torch.library.wrap_triton(_grad_vertices_kernel))
    kernel[(triton.cdiv(total, _BLOCK_SIZE),)](
        grad_output,
        transforms,
        joint_indices,
        joint_weights,
        grad_vertices,
        total,
        num_vertices=grad_output.shape[1],
        num_joints=transforms.shape[1],
        num_slots=joint_indices.shape[1],
        block_size=_BLOCK_SIZE,
    )
    return grad_vertices


@torch.library.triton_op("body_models::compact_linear_blend_skinning_grad_transforms", mutates_args={})
def _compact_lbs_grad_transforms_op(
    grad_output: Tensor,
    vertices: Tensor,
    transforms: Tensor,
    plan_vertex_indices: Tensor,
    plan_weights: Tensor,
    plan_offsets: Tensor,
) -> Tensor:
    grad_transforms = torch.zeros_like(transforms)
    kernel = cast(Any, torch.library.wrap_triton(_grad_transforms_kernel))
    kernel[(grad_output.shape[0] * transforms.shape[1],)](
        grad_output,
        vertices,
        plan_vertex_indices,
        plan_weights,
        plan_offsets,
        grad_transforms,
        num_vertices=vertices.shape[1],
        num_joints=transforms.shape[1],
        plan_joints=plan_offsets.shape[0] - 1,
        block_size=_BLOCK_SIZE,
        num_warps=_TRANSFORM_NUM_WARPS,
    )
    return grad_transforms


def _setup_context(ctx: Any, inputs: tuple[Tensor, ...], output: Tensor) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
    vertices, transforms, joint_indices, joint_weights, plan_indices, plan_weights, plan_offsets = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    grad_vertices = None
    grad_transforms = None
    if ctx.needs_input_grad[0]:
        grad_vertices = _compact_lbs_grad_vertices_op(grad_output, transforms, joint_indices, joint_weights)
    if ctx.needs_input_grad[1]:
        grad_transforms = _compact_lbs_grad_transforms_op(
            grad_output,
            vertices,
            transforms,
            plan_indices,
            plan_weights,
            plan_offsets,
        )
    return grad_vertices, grad_transforms, None, None, None, None, None


_compact_lbs_op.register_autograd(_backward, setup_context=_setup_context)


def _build_transform_gradient_plan(joint_indices: Tensor, joint_weights: Tensor) -> _JointMajorCsrPlan:
    flat_indices = joint_indices.reshape(-1)
    flat_weights = joint_weights.reshape(-1)
    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).flatten()
    valid_joints = flat_indices[valid_positions]
    permutation = torch.argsort(valid_joints)
    sorted_positions = valid_positions[permutation]
    num_joints = int(valid_joints.max().item()) + 1
    counts = torch.bincount(valid_joints.to(torch.int64), minlength=num_joints)
    offsets = torch.nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))
    num_slots = joint_indices.shape[1]
    return _JointMajorCsrPlan(
        vertex_indices=torch.div(sorted_positions, num_slots, rounding_mode="floor").to(torch.int32).contiguous(),
        weights=flat_weights[sorted_positions].contiguous(),
        offsets=offsets.to(torch.int32).contiguous(),
    )


def _rebuild_transform_gradient_plan(module: TritonSkinningState, incompatible_keys: object) -> None:
    del incompatible_keys
    plan = _build_transform_gradient_plan(module.joint_indices, module.joint_weights)
    module._plan_vertex_indices = plan.vertex_indices
    module._plan_weights = plan.weights
    module._plan_offsets = plan.offsets
