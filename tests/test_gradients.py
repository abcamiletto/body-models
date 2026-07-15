import numpy as np
import pytest

import model_cases
from body_models.base import RigidBodyModel


def surface_loss(model, params):
    if isinstance(model, RigidBodyModel):
        return model.forward_links(**params)[..., :1, :3, 3].sum()
    return model.forward_vertices(**params)[..., :8, :].sum()


@pytest.mark.parametrize(("name", "_numpy_model", "torch_model", "jax_model", "kwargs"), model_cases.MODELS)
def test_torch_and_jax_gradients_match_finite_difference(name, _numpy_model, torch_model, jax_model, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_instance = torch_model(**kwargs)
    torch_instance.double()
    torch_params = torch_instance.get_rest_pose(batch_dims=(), dtype=torch.float64)
    torch_keys = [key for key, value in torch_params.items() if np.asarray(value).size]
    if "global_translation" in torch_keys:
        torch_keys.remove("global_translation")
        torch_keys.insert(0, "global_translation")
    for torch_key in torch_keys:
        torch_value = torch_params[torch_key].clone().requires_grad_(True)
        torch_params[torch_key] = torch_value
        torch_loss_value = surface_loss(torch_instance, torch_params)
        if torch_loss_value.requires_grad:
            break
        torch_params[torch_key] = torch_value.detach()
    torch_loss_value.backward()

    torch_auto = torch_value.grad.reshape(-1)[0].item()
    torch_plus = torch_value.detach().numpy().copy()
    torch_minus = torch_value.detach().numpy().copy()
    torch_plus.reshape(-1)[0] += 1e-4
    torch_minus.reshape(-1)[0] -= 1e-4
    plus_params = torch_params.copy()
    minus_params = torch_params.copy()
    plus_params[torch_key] = torch.as_tensor(torch_plus, dtype=torch_value.dtype)
    minus_params[torch_key] = torch.as_tensor(torch_minus, dtype=torch_value.dtype)
    with torch.no_grad():
        torch_plus_loss = surface_loss(torch_instance, plus_params).item()
        torch_minus_loss = surface_loss(torch_instance, minus_params).item()
    torch_numeric = (torch_plus_loss - torch_minus_loss) / 2e-4
    np.testing.assert_allclose(torch_auto, torch_numeric, rtol=1e-2, atol=1e-2)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    jax_instance = jax_model(**kwargs)
    jax_params = jax_instance.get_rest_pose(batch_dims=(), dtype=jnp.float64)
    jax_key = torch_key
    jax_value = jax_params[jax_key]

    def jax_loss(value):
        params = jax_params.copy()
        params[jax_key] = value
        return surface_loss(jax_instance, params)

    jax_auto = np.asarray(jax.grad(jax_loss)(jax_value)).reshape(-1)[0]
    jax_plus = np.asarray(jax_value).copy()
    jax_minus = np.asarray(jax_value).copy()
    jax_plus.reshape(-1)[0] += 1e-4
    jax_minus.reshape(-1)[0] -= 1e-4
    jax_numeric = (float(jax_loss(jnp.asarray(jax_plus))) - float(jax_loss(jnp.asarray(jax_minus)))) / 2e-4
    np.testing.assert_allclose(jax_auto, jax_numeric, rtol=1e-2, atol=1e-2)


@pytest.mark.fast
def test_warp_skinning_gradients_match_torch_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")

    from body_models.common import skinning
    from body_models.common import warp as warp_backend

    torch.manual_seed(42)
    num_batches, num_vertices, num_joints, num_slots = 2, 257, 31, 6
    joint_indices = torch.randint(
        num_joints,
        (num_vertices, num_slots),
        dtype=torch.int32,
    )
    joint_weights = torch.rand(num_vertices, num_slots)
    joint_weights /= joint_weights.sum(dim=-1, keepdim=True)
    dense_weights = torch.zeros(num_vertices, num_joints)
    dense_weights.scatter_add_(1, joint_indices.long(), joint_weights)

    vertices = torch.randn(1, num_vertices, 3, requires_grad=True)
    transforms = torch.randn(num_batches, num_joints, 4, 4, requires_grad=True)
    grad_output = torch.randn(num_batches, num_vertices, 3)

    expected = skinning.linear_blend_skinning(vertices, transforms, dense_weights, xp=torch)
    expected_grads = torch.autograd.grad(expected, (vertices, transforms), grad_output)
    actual = warp_backend.compact_linear_blend_skinning(
        vertices,
        transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )
    actual_grads = torch.autograd.grad(actual, (vertices, transforms), grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.fast
def test_warp_forward_kinematics_gradients_match_common_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")

    from body_models.common import kinematics
    from body_models.common import warp as warp_backend

    torch.manual_seed(17)
    parents = [-1, 0, 1, 1]
    rotations = torch.randn(2, 4, 3, 3, requires_grad=True)
    translations = torch.randn(2, 4, 3, requires_grad=True)
    grad_output = torch.randn(2, 4, 4, 4)

    fronts = kinematics.compute_kinematic_fronts(parents)
    expected = kinematics.forward_kinematics(rotations, translations, fronts, xp=torch)
    expected_grads = torch.autograd.grad(expected, (rotations, translations), grad_output)
    actual = warp_backend.forward_kinematics(rotations, translations, torch.as_tensor(parents))
    actual_grads = torch.autograd.grad(actual, (rotations, translations), grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.fast
def test_warp_runtime_matches_common_skinning() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")

    from body_models.common import skinning
    from body_models.runtime import TorchRuntime

    joint_indices = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int32)
    joint_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.8, 0.2]])
    dense_weights = torch.zeros(3, 3)
    dense_weights.scatter_add_(1, joint_indices.long(), joint_weights)
    rest_vertices = torch.randn(2, 3, 3)
    pose_offsets = torch.randn(2, 3, 3)
    transforms = torch.randn(2, 3, 4, 4)

    expected = skinning.linear_blend_skinning(rest_vertices + pose_offsets, transforms, dense_weights, xp=torch)
    actual = TorchRuntime("warp").compact_linear_blend_skinning(
        rest_vertices + pose_offsets,
        transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_soma_warp_forward_and_gradients_match_torch() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("SOMA's Warp kernel requires CUDA")

    from body_models.bodies.soma import torch as soma_torch

    torch.manual_seed(7)
    models = {kernel: soma_torch.SOMA(kernel=kernel).cuda() for kernel in ("torch", "warp")}
    params = models["torch"].get_rest_pose(batch_dims=(1,))
    params = {key: value + 0.01 * torch.randn_like(value) for key, value in params.items()}
    grad_output = torch.randn(1, models["torch"].num_vertices, 3, device="cuda")
    param_keys = tuple(params)
    results = {}

    for kernel, model in models.items():
        kernel_params = {key: value.detach().requires_grad_(True) for key, value in params.items()}
        vertices = model.forward_vertices(**kernel_params)
        grads = torch.autograd.grad(vertices, tuple(kernel_params.values()), grad_output)
        results[kernel] = vertices, dict(zip(param_keys, grads, strict=True))

    torch_vertices, torch_grads = results["torch"]
    warp_vertices, warp_grads = results["warp"]
    torch.testing.assert_close(warp_vertices, torch_vertices, rtol=1e-5, atol=1e-5)
    for key in torch_grads:
        torch.testing.assert_close(warp_grads[key], torch_grads[key], rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize(
    ("name", "_numpy_model", "torch_model", "_jax_model", "kwargs"),
    [case for case in model_cases.SKINNED_MODELS if case[0] == "garment_measurements"],
)
def test_torch_kernel_gradients_match_default(name, _numpy_model, torch_model, _jax_model, kwargs) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    default_model = torch_model(**kwargs).cuda()
    params = default_model.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    vertex_indices = list(range(min(8, default_model.num_vertices)))
    generator = torch.Generator(device="cuda").manual_seed(0)
    params = {
        key: value + 0.1 * torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
        for key, value in params.items()
    }

    def forward_and_grad(model):
        model_params = {key: value.detach().clone().requires_grad_() for key, value in params.items()}
        vertices = model.forward_vertices(**model_params, vertex_indices=vertex_indices)
        gradients = torch.autograd.grad(vertices.square().sum(), tuple(model_params.values()))
        return vertices, gradients

    expected_vertices, expected_gradients = forward_and_grad(default_model)
    for kernel in default_model.kernels[1:]:
        actual_vertices, actual_gradients = forward_and_grad(torch_model(kernel=kernel, **kwargs).cuda())
        torch.testing.assert_close(actual_vertices, expected_vertices, rtol=1e-4, atol=1e-4)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
