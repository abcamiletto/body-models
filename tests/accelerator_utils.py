"""Shared utilities for accelerator-aware tests."""

import torch


def get_accelerator_device() -> torch.device | None:
    """Return the best available torch accelerator device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return None
