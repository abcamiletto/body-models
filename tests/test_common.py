"""Tests for shared backend utilities."""

import numpy as np
import pytest

from body_models import common

pytestmark = pytest.mark.fast


def test_get_namespace_numpy() -> None:
    assert common.get_namespace(np.ones(1)) is np


def test_get_namespace_torch() -> None:
    torch = pytest.importorskip("torch")

    assert common.get_namespace(torch.ones(1)) is torch


def test_get_namespace_jax() -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    assert common.get_namespace(jnp.ones(1)) is jnp
