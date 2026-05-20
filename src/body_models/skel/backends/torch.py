"""PyTorch SKEL backend."""

import torch

from body_models.skel.backends.core import forward_skeleton as _forward_skeleton
from body_models.skel.backends.core import forward_vertices as _forward_vertices
from body_models.skel.backends.core import prepare_identity as _prepare_identity

__all__ = ["forward_skeleton", "forward_vertices", "prepare_identity"]


def prepare_identity(*args, **kwargs):
    return _prepare_identity(*args, **kwargs, xp=torch)


def forward_vertices(*args, **kwargs):
    return _forward_vertices(*args, **kwargs, xp=torch)


def forward_skeleton(*args, **kwargs):
    return _forward_skeleton(*args, **kwargs, xp=torch)
