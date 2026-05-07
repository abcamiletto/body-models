"""JAX SKEL backend."""

import jax.numpy as jnp

from body_models.skel.backends.core import forward_skeleton as _forward_skeleton
from body_models.skel.backends.core import forward_vertices as _forward_vertices

__all__ = ["forward_skeleton", "forward_vertices"]


def forward_vertices(*args, **kwargs):
    return _forward_vertices(*args, **kwargs, xp=jnp)


def forward_skeleton(*args, **kwargs):
    return _forward_skeleton(*args, **kwargs, xp=jnp)
