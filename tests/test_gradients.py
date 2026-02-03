"""Test gradient correctness for all body models.

Uses torch.autograd.gradcheck to verify analytical gradients match numerical gradients.
"""

import pytest
import torch
from torch.autograd import gradcheck

from body_models import ANNY, MHR, SKEL, SMPL, SMPLX


def _prepare_params(params: dict[str, torch.Tensor], requires_grad: bool = True) -> dict[str, torch.Tensor]:
    """Convert params to float64 and optionally enable gradients."""
    result = {}
    for k, v in params.items():
        if v.dtype.is_floating_point:
            v = v.to(torch.float64)
            if requires_grad:
                v = v.requires_grad_(True)
        result[k] = v
    return result


def _make_gradcheck_fn(model, method_name: str):
    """Create a function suitable for gradcheck from a model method."""
    method = getattr(model, method_name)

    def fn(*tensors, param_names):
        kwargs = dict(zip(param_names, tensors))
        return method(**kwargs)

    return fn


class TestSMPLGradients:
    @pytest.fixture
    def model(self):
        model = SMPL()
        return model.to(torch.float64).eval()

    def test_forward_vertices_gradients(self, model):
        """Test gradients flow correctly through forward_vertices."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        # Add small noise to avoid zero gradients at rest pose
        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_vertices(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)

    def test_forward_skeleton_gradients(self, model):
        """Test gradients flow correctly through forward_skeleton."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_skeleton(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)


class TestSMPLXGradients:
    @pytest.fixture
    def model(self):
        model = SMPLX()
        return model.to(torch.float64).eval()

    def test_forward_vertices_gradients(self, model):
        """Test gradients flow correctly through forward_vertices."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_vertices(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)

    def test_forward_skeleton_gradients(self, model):
        """Test gradients flow correctly through forward_skeleton."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_skeleton(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)


class TestSKELGradients:
    @pytest.fixture
    def model(self):
        model = SKEL(gender="male")
        return model.to(torch.float64).eval()

    def test_forward_vertices_gradients(self, model):
        """Test gradients flow correctly through forward_vertices."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_vertices(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)

    def test_forward_skeleton_gradients(self, model):
        """Test gradients flow correctly through forward_skeleton."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_skeleton(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)


class TestANNYGradients:
    @pytest.fixture
    def model(self):
        model = ANNY()
        return model.to(torch.float64).eval()

    def test_forward_vertices_gradients(self, model):
        """Test gradients flow correctly through forward_vertices."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_vertices(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)

    def test_forward_skeleton_gradients(self, model):
        """Test gradients flow correctly through forward_skeleton."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_skeleton(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)


class TestMHRGradients:
    @pytest.fixture
    def model(self):
        model = MHR()
        return model.to(torch.float64).eval()

    def test_forward_vertices_gradients(self, model):
        """Test gradients flow correctly through forward_vertices."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_vertices(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)

    def test_forward_skeleton_gradients(self, model):
        """Test gradients flow correctly through forward_skeleton."""
        params = _prepare_params(model.get_rest_pose(batch_size=2))

        for k, v in params.items():
            if v.requires_grad:
                params[k] = v + torch.randn_like(v) * 0.01

        def fn(*tensors):
            kwargs = dict(zip(params.keys(), tensors))
            return model.forward_skeleton(**kwargs)

        inputs = tuple(params.values())
        assert gradcheck(fn, inputs, raise_exception=True, fast_mode=True)
