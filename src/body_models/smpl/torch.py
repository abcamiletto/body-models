"""PyTorch backend for SMPL model."""

from ._torch_impl import SMPL, from_native_args, to_native_outputs

__all__ = ["SMPL", "from_native_args", "to_native_outputs"]
