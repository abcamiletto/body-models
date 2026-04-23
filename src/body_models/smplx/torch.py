"""PyTorch backend for SMPL-X model."""

from ._torch_impl import SMPLX, from_native_args, to_native_outputs

__all__ = ["SMPLX", "from_native_args", "to_native_outputs"]
