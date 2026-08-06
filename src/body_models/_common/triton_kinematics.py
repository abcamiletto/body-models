"""Triton lowering for parent-tree affine transform composition."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int
from torch import Tensor, nn

from body_models._common.kinematics import Front, KinematicTree

__all__ = ["TritonKinematicTree", "compose_parent_tree", "prepare_kinematic_tree"]

_BLOCK_SIZE = 64


class TritonKinematicTree(nn.Module):
    """Immutable tree metadata with a materialized device parent table."""

    parents: tuple[int, ...]
    fronts: tuple[Front, ...]
    parent_indices: Int[Tensor, "J"]

    def __init__(self, tree: KinematicTree) -> None:
        super().__init__()
        if any(parent > joint for joint, parent in enumerate(tree.parents)):
            raise ValueError("Triton kinematics requires parents to precede their children")

        normalized = [joint if parent < 0 else parent for joint, parent in enumerate(tree.parents)]
        self.parents = tree.parents
        self.fronts = tree.fronts
        self.register_buffer(
            "parent_indices",
            torch.tensor(normalized, dtype=torch.int32),
            persistent=False,
        )


def prepare_kinematic_tree(tree: KinematicTree) -> TritonKinematicTree:
    """Materialize a validated parent table for Triton composition."""
    return TritonKinematicTree(tree)


@triton.jit
def _matmul3(a, b):
    return (
        a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
        a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
        a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
        a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
        a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
        a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
        a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
        a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
        a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    )


@triton.jit
def _matvec3(a, v):
    return (
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    )


@triton.jit
def _transpose3(a):
    return a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]


@triton.jit
def _load_affine(transforms, base, mask):
    return (
        tl.load(transforms + base, mask=mask),
        tl.load(transforms + base + 1, mask=mask),
        tl.load(transforms + base + 2, mask=mask),
        tl.load(transforms + base + 4, mask=mask),
        tl.load(transforms + base + 5, mask=mask),
        tl.load(transforms + base + 6, mask=mask),
        tl.load(transforms + base + 8, mask=mask),
        tl.load(transforms + base + 9, mask=mask),
        tl.load(transforms + base + 10, mask=mask),
    ), (
        tl.load(transforms + base + 3, mask=mask),
        tl.load(transforms + base + 7, mask=mask),
        tl.load(transforms + base + 11, mask=mask),
    )


@triton.jit
def _store_upper(output, base, linear, translation, mask):
    tl.store(output + base, linear[0], mask=mask)
    tl.store(output + base + 1, linear[1], mask=mask)
    tl.store(output + base + 2, linear[2], mask=mask)
    tl.store(output + base + 3, translation[0], mask=mask)
    tl.store(output + base + 4, linear[3], mask=mask)
    tl.store(output + base + 5, linear[4], mask=mask)
    tl.store(output + base + 6, linear[5], mask=mask)
    tl.store(output + base + 7, translation[1], mask=mask)
    tl.store(output + base + 8, linear[6], mask=mask)
    tl.store(output + base + 9, linear[7], mask=mask)
    tl.store(output + base + 10, linear[8], mask=mask)
    tl.store(output + base + 11, translation[2], mask=mask)


@triton.jit
def _store_affine(output, base, linear, translation, bottom_right, mask):
    _store_upper(output, base, linear, translation, mask)
    tl.store(output + base + 12, 0.0, mask=mask)
    tl.store(output + base + 13, 0.0, mask=mask)
    tl.store(output + base + 14, 0.0, mask=mask)
    tl.store(output + base + 15, bottom_right, mask=mask)


@triton.jit
def _compose_kernel(local, parents, world, num_batches, num_joints, BLOCK_SIZE: tl.constexpr):
    batch = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch < num_batches
    # This runtime loop keeps generated code size independent of joint count.
    for joint in tl.range(0, num_joints, loop_unroll_factor=1):
        base = (batch * num_joints + joint) * 16
        local_linear, local_translation = _load_affine(local, base, mask)
        parent = tl.load(parents + joint)
        is_root = parent == joint
        parent_base = (batch * num_joints + parent) * 16
        parent_linear, parent_translation = _load_affine(world, parent_base, mask)
        composed_linear = _matmul3(parent_linear, local_linear)
        rotated_translation = _matvec3(parent_linear, local_translation)
        composed_translation = (
            parent_translation[0] + rotated_translation[0],
            parent_translation[1] + rotated_translation[1],
            parent_translation[2] + rotated_translation[2],
        )
        world_linear = (
            tl.where(is_root, local_linear[0], composed_linear[0]),
            tl.where(is_root, local_linear[1], composed_linear[1]),
            tl.where(is_root, local_linear[2], composed_linear[2]),
            tl.where(is_root, local_linear[3], composed_linear[3]),
            tl.where(is_root, local_linear[4], composed_linear[4]),
            tl.where(is_root, local_linear[5], composed_linear[5]),
            tl.where(is_root, local_linear[6], composed_linear[6]),
            tl.where(is_root, local_linear[7], composed_linear[7]),
            tl.where(is_root, local_linear[8], composed_linear[8]),
        )
        world_translation = (
            tl.where(is_root, local_translation[0], composed_translation[0]),
            tl.where(is_root, local_translation[1], composed_translation[1]),
            tl.where(is_root, local_translation[2], composed_translation[2]),
        )
        _store_affine(world, base, world_linear, world_translation, 1.0, mask)


@triton.jit
def _compose_backward_kernel(
    grad_world,
    local,
    parents,
    world,
    grad_local,
    num_batches,
    num_joints,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch < num_batches
    # Reverse runtime traversal avoids unrolling one VJP per joint.
    for offset in tl.range(0, num_joints, loop_unroll_factor=1):
        joint = num_joints - offset - 1
        base = (batch * num_joints + joint) * 16
        output_linear, output_translation = _load_affine(grad_world, base, mask)
        parent = tl.load(parents + joint)
        is_root = parent == joint
        parent_base = (batch * num_joints + parent) * 16
        parent_linear, _ = _load_affine(world, parent_base, mask)
        input_linear, input_translation = _load_affine(local, base, mask)
        parent_transpose = _transpose3(parent_linear)
        composed_local_linear = _matmul3(parent_transpose, output_linear)
        composed_local_translation = _matvec3(parent_transpose, output_translation)
        local_linear = (
            tl.where(is_root, output_linear[0], composed_local_linear[0]),
            tl.where(is_root, output_linear[1], composed_local_linear[1]),
            tl.where(is_root, output_linear[2], composed_local_linear[2]),
            tl.where(is_root, output_linear[3], composed_local_linear[3]),
            tl.where(is_root, output_linear[4], composed_local_linear[4]),
            tl.where(is_root, output_linear[5], composed_local_linear[5]),
            tl.where(is_root, output_linear[6], composed_local_linear[6]),
            tl.where(is_root, output_linear[7], composed_local_linear[7]),
            tl.where(is_root, output_linear[8], composed_local_linear[8]),
        )
        local_translation = (
            tl.where(is_root, output_translation[0], composed_local_translation[0]),
            tl.where(is_root, output_translation[1], composed_local_translation[1]),
            tl.where(is_root, output_translation[2], composed_local_translation[2]),
        )
        _store_affine(grad_local, base, local_linear, local_translation, 0.0, mask)

        parent_linear_gradient = _matmul3(output_linear, _transpose3(input_linear))
        parent_linear_gradient = (
            parent_linear_gradient[0] + output_translation[0] * input_translation[0],
            parent_linear_gradient[1] + output_translation[0] * input_translation[1],
            parent_linear_gradient[2] + output_translation[0] * input_translation[2],
            parent_linear_gradient[3] + output_translation[1] * input_translation[0],
            parent_linear_gradient[4] + output_translation[1] * input_translation[1],
            parent_linear_gradient[5] + output_translation[1] * input_translation[2],
            parent_linear_gradient[6] + output_translation[2] * input_translation[0],
            parent_linear_gradient[7] + output_translation[2] * input_translation[1],
            parent_linear_gradient[8] + output_translation[2] * input_translation[2],
        )
        current_linear, current_translation = _load_affine(grad_world, parent_base, mask)
        parent_linear_gradient = (
            current_linear[0] + parent_linear_gradient[0],
            current_linear[1] + parent_linear_gradient[1],
            current_linear[2] + parent_linear_gradient[2],
            current_linear[3] + parent_linear_gradient[3],
            current_linear[4] + parent_linear_gradient[4],
            current_linear[5] + parent_linear_gradient[5],
            current_linear[6] + parent_linear_gradient[6],
            current_linear[7] + parent_linear_gradient[7],
            current_linear[8] + parent_linear_gradient[8],
        )
        parent_translation_gradient = (
            current_translation[0] + output_translation[0],
            current_translation[1] + output_translation[1],
            current_translation[2] + output_translation[2],
        )
        _store_upper(
            grad_world,
            parent_base,
            parent_linear_gradient,
            parent_translation_gradient,
            mask & ~is_root,
        )


@torch.library.triton_op("body_models::compose_parent_tree", mutates_args={})
def _compose_parent_tree_op(local: Tensor, parents: Tensor) -> Tensor:
    world = torch.empty_like(local)
    kernel = cast(Any, torch.library.wrap_triton(_compose_kernel))
    kernel[(triton.cdiv(local.shape[0], _BLOCK_SIZE),)](
        local,
        parents,
        world,
        local.shape[0],
        local.shape[1],
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=2,
    )
    return world


@torch.library.triton_op("body_models::compose_parent_tree_backward", mutates_args={})
def _compose_parent_tree_backward_op(
    grad_output: Tensor,
    local: Tensor,
    parents: Tensor,
    world: Tensor,
) -> Tensor:
    grad_world = grad_output.contiguous().clone()
    grad_local = torch.empty_like(local)
    kernel = cast(Any, torch.library.wrap_triton(_compose_backward_kernel))
    kernel[(triton.cdiv(local.shape[0], _BLOCK_SIZE),)](
        grad_world,
        local,
        parents,
        world,
        grad_local,
        local.shape[0],
        local.shape[1],
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=2,
    )
    return grad_local


def _setup_context(ctx: Any, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
    local, parents = inputs
    ctx.save_for_backward(local, parents, output)


def _backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, None]:
    if not ctx.needs_input_grad[0]:
        return None, None
    local, parents, world = ctx.saved_tensors
    grad_local = _compose_parent_tree_backward_op(grad_output, local, parents, world)
    return grad_local, None


_compose_parent_tree_op.register_autograd(_backward, setup_context=_setup_context)


def compose_parent_tree(
    local: Float[Tensor, "*batch J 4 4"],
    parents: Int[Tensor, "J"],
) -> Float[Tensor, "*batch J 4 4"]:
    """Compose affine transforms whose bottom rows are ``[0, 0, 0, 1]``."""
    if not local.is_cuda or not parents.is_cuda:
        raise TypeError("Triton kinematics requires all tensors on CUDA")
    if local.device != parents.device:
        raise TypeError("Triton kinematics requires all tensors on the same CUDA device")
    if local.dtype != torch.float32:
        raise TypeError("Triton kinematics requires float32 tensors")
    if parents.dtype != torch.int32:
        raise TypeError("Triton kinematics requires int32 parent indices")
    if local.shape[-2:] != (4, 4):
        raise ValueError("local transforms must have shape [..., J, 4, 4]")
    if parents.ndim != 1 or parents.shape[0] != local.shape[-3]:
        raise ValueError("parents must contain one entry per joint")

    batch_shape = local.shape[:-3]
    num_joints = local.shape[-3]
    flat_local = local.reshape(-1, num_joints, 4, 4).contiguous()
    world = _compose_parent_tree_op(flat_local, parents)
    return world.reshape(*batch_shape, num_joints, 4, 4)
