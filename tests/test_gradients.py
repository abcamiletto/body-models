import model_cases
import numpy as np
import pytest

from body_models import RigidBodyModel, TorchRuntime


def surface_loss(model, params):
    if isinstance(model, RigidBodyModel):
        values = model.forward_links(**params)[..., :3, 3]
    else:
        values = model.forward_vertices(**params)
    return (values**2).sum()


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_torch_and_jax_gradients_match_finite_difference(name, model_class, kwargs) -> None:
    torch = pytest.importorskip("torch")
    torch_class = model_cases.backend_model_class(name, "torch")
    torch_instance = torch_class(**kwargs).double()
    torch_rest = torch_instance.get_rest_pose(batch_dims=(), dtype=torch.float64)
    torch_rest = {key: value + 0.03 for key, value in torch_rest.items()}

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    jax_class = model_cases.backend_model_class(name, "jax")
    jax_instance = jax_class(**kwargs)
    jax_rest = jax_instance.get_rest_pose(batch_dims=(), dtype=jnp.float64)
    jax_rest = {key: value + 0.03 for key, value in jax_rest.items()}

    for key in torch_rest:
        torch_params = {name: value.detach() for name, value in torch_rest.items()}
        torch_value = torch_params[key].clone().requires_grad_(True)
        torch_params[key] = torch_value
        torch_loss_value = surface_loss(torch_instance, torch_params)
        assert torch_loss_value.requires_grad, f"{name}.{key} is disconnected from the Torch output"
        torch_loss_value.backward()

        torch_auto = torch_value.grad.reshape(-1)[0].item()
        torch_plus = torch_value.detach().numpy().copy()
        torch_minus = torch_value.detach().numpy().copy()
        torch_plus.reshape(-1)[0] += 1e-4
        torch_minus.reshape(-1)[0] -= 1e-4
        plus_params = torch_params.copy()
        minus_params = torch_params.copy()
        plus_params[key] = torch.as_tensor(torch_plus, dtype=torch_value.dtype)
        minus_params[key] = torch.as_tensor(torch_minus, dtype=torch_value.dtype)
        with torch.no_grad():
            torch_plus_loss = surface_loss(torch_instance, plus_params).item()
            torch_minus_loss = surface_loss(torch_instance, minus_params).item()
        torch_numeric = (torch_plus_loss - torch_minus_loss) / 2e-4
        np.testing.assert_allclose(
            torch_auto,
            torch_numeric,
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"Torch gradient mismatch for {name}.{key}",
        )

        jax_value = jax_rest[key]

        def jax_loss(value, parameter=key):
            params = jax_rest.copy()
            params[parameter] = value
            return surface_loss(jax_instance, params)

        jax_auto = np.asarray(jax.grad(jax_loss)(jax_value)).reshape(-1)[0]
        jax_plus = np.asarray(jax_value).copy()
        jax_minus = np.asarray(jax_value).copy()
        jax_plus.reshape(-1)[0] += 1e-4
        jax_minus.reshape(-1)[0] -= 1e-4
        jax_numeric = (float(jax_loss(jnp.asarray(jax_plus))) - float(jax_loss(jnp.asarray(jax_minus)))) / 2e-4
        np.testing.assert_allclose(
            jax_auto,
            jax_numeric,
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"JAX gradient mismatch for {name}.{key}",
        )


@pytest.mark.fast
def test_compact_and_warp_skinning_gradients_match_dense_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")

    from body_models._common import skinning

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
    compact = skinning.compact_linear_blend_skinning(
        vertices,
        transforms,
        joint_indices=joint_indices,
        joint_weights=joint_weights,
        xp=torch,
    )
    compact_grads = torch.autograd.grad(compact, (vertices, transforms), grad_output)
    warp_runtime = TorchRuntime(skinning_backend="warp")
    warp_skinning = warp_runtime._materialize(skinning.CompactSkinning(joint_indices, joint_weights))
    warp = warp_runtime._skin_vertices(
        vertices,
        transforms,
        skinning=warp_skinning,
    )
    warp_grads = torch.autograd.grad(warp, (vertices, transforms), grad_output)

    for actual, actual_grads in ((compact, compact_grads), (warp, warp_grads)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_soma_warp_forward_and_gradients_match_torch() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("SOMA's Warp skinning backend requires CUDA")

    from body_models.soma.torch import SOMA

    torch.manual_seed(7)
    models = {
        skinning_backend: SOMA(skinning_backend=skinning_backend).cuda() for skinning_backend in ("torch", "warp")
    }
    params = models["torch"].get_rest_pose(batch_dims=(1,))
    params = {key: value + 0.01 * torch.randn_like(value) for key, value in params.items()}
    grad_output = torch.randn(1, models["torch"].num_vertices, 3, device="cuda")
    param_keys = tuple(params)
    results = {}

    for skinning_backend, model in models.items():
        backend_params = {key: value.detach().requires_grad_(True) for key, value in params.items()}
        vertices = model.forward_vertices(**backend_params)
        grads = torch.autograd.grad(vertices, tuple(backend_params.values()), grad_output)
        results[skinning_backend] = vertices, dict(zip(param_keys, grads, strict=True))

    torch_vertices, torch_grads = results["torch"]
    warp_vertices, warp_grads = results["warp"]
    torch.testing.assert_close(warp_vertices, torch_vertices, rtol=1e-5, atol=1e-5)
    for key in torch_grads:
        torch.testing.assert_close(warp_grads[key], torch_grads[key], rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize(
    ("name", "model_class", "kwargs"),
    [case for case in model_cases.SKINNED_MODELS if case[0] == "garment_measurements"],
)
def test_torch_skinning_backend_gradients_match_default(
    name,
    model_class,
    kwargs,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch_class = model_cases.backend_model_class(name, "torch")
    default_model = torch_class(**kwargs).cuda()
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
    for skinning_backend in TorchRuntime.SKINNING_BACKENDS[1:]:
        model = torch_class(**kwargs, skinning_backend=skinning_backend).cuda()
        actual_vertices, actual_gradients = forward_and_grad(model)
        torch.testing.assert_close(actual_vertices, expected_vertices, rtol=1e-4, atol=1e-4)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
