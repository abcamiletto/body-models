import model_cases
import numpy as np
import pytest

from body_models._rotations import VALID_ROTATION_TYPES
from body_models._runtime import TorchRuntime


def surface_loss(model, params):
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
    warp_runtime = TorchRuntime(kernel_backend="warp")
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
def test_triton_skinning_gradients_match_dense_under_compile() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from body_models._common import skinning

    torch.manual_seed(42)
    num_batches, num_vertices, num_joints, num_slots = 2, 257, 31, 6
    joint_indices = torch.randint(num_joints - 1, (num_vertices, num_slots), dtype=torch.int32, device="cuda")
    joint_weights = torch.rand(num_vertices, num_slots, device="cuda")
    joint_indices[:, -1] = -1
    joint_weights[:, -1] = 0
    joint_weights /= joint_weights.sum(dim=-1, keepdim=True)
    dense_weights = torch.zeros(num_vertices, num_joints, device="cuda")
    dense_weights.scatter_add_(1, joint_indices.clamp_min(0).long(), joint_weights)
    vertices = torch.randn(1, num_vertices, 3, device="cuda", requires_grad=True)
    transforms = torch.randn(num_batches, num_joints, 4, 4, device="cuda", requires_grad=True)
    grad_output = torch.randn(num_batches, num_vertices, 3, device="cuda")

    expected = skinning.linear_blend_skinning(vertices, transforms, dense_weights, xp=torch)
    expected_gradients = torch.autograd.grad(expected, (vertices, transforms), grad_output)
    runtime = TorchRuntime(kernel_backend="triton")
    state = runtime._materialize(skinning.CompactSkinning(joint_indices, joint_weights))
    with pytest.raises(TypeError, match="all tensors on CUDA"):
        runtime._skin_vertices(vertices.cpu(), transforms.cpu(), skinning=state.cpu())
    state.cuda()
    with pytest.raises(TypeError, match="float32"):
        runtime._skin_vertices(vertices.double(), transforms.double(), skinning=state)
    forward = torch.compile(lambda v, t: runtime._skin_vertices(v, t, skinning=state), fullgraph=True)
    actual = forward(vertices, transforms)
    actual_gradients = torch.autograd.grad(actual, (vertices, transforms), grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("num_joints", "batch_size", "multiple_roots"),
    [(8, 2, True), (163, 129, False)],
)
def test_triton_kinematics_matches_torch_under_compile(num_joints, batch_size, multiple_roots) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from body_models._common import kinematics, triton_kinematics

    torch.manual_seed(7)
    parents = [0] + [(joint - 1) // 2 for joint in range(1, num_joints)]
    if multiple_roots:
        parents[num_joints // 2] = -1
    tree = kinematics.KinematicTree.from_parents(parents)
    triton_tree = triton_kinematics.prepare_kinematic_tree(tree).cuda()
    linear = torch.eye(3, device="cuda").expand(batch_size, num_joints, 3, 3)
    linear = linear + 0.01 * torch.randn_like(linear)
    translation = torch.randn(batch_size, num_joints, 3, 1, device="cuda")
    upper = torch.cat((linear, translation), dim=-1).requires_grad_()
    reference_upper = upper.detach().double().cpu().requires_grad_()

    def affine(value):
        bottom = torch.zeros(*value.shape[:-2], 1, 4, dtype=value.dtype, device=value.device)
        bottom[..., 0, 3] = 1
        return torch.cat((value, bottom), dim=-2)

    expected = kinematics.compose_kinematic_fronts(affine(reference_upper), tree.fronts, xp=torch)
    forward = torch.compile(
        lambda value: triton_kinematics.compose_parent_tree(affine(value), triton_tree.parent_indices),
        fullgraph=True,
    )
    actual = forward(upper)
    grad_output = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, upper, grad_output)[0]
    expected_gradient = torch.autograd.grad(expected, reference_upper, grad_output.double().cpu())[0]

    torch.testing.assert_close(actual.double().cpu(), expected, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_gradient.double().cpu(), expected_gradient, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("device", "dtype_name", "error"),
    [
        ("cpu", "float32", "CUDA"),
        ("cpu", "float64", "CUDA"),
        ("cuda", "float64", "float32"),
    ],
)
def test_triton_model_rejects_unsupported_kinematics(device, dtype_name, error) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from body_models.smpl.torch import SMPL

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dtype = getattr(torch, dtype_name)
    model = SMPL(gender="neutral", kernel_backend="triton").to(device=device, dtype=dtype)
    params = model.get_rest_pose(batch_dims=(2,), dtype=dtype)

    with pytest.raises(TypeError, match=error):
        model.forward_skeleton(**params)


@pytest.mark.slow
@pytest.mark.parametrize("rotation_type", VALID_ROTATION_TYPES)
def test_triton_kinematics_supports_all_rotation_representations(rotation_type) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from body_models.smpl.torch import SMPL

    torch.manual_seed(11)
    reference = SMPL(gender="neutral", rotation_type=rotation_type).cuda()
    accelerated = SMPL(gender="neutral", rotation_type=rotation_type, kernel_backend="triton").cuda()
    params = reference.get_rest_pose(batch_dims=(2,), dtype=torch.float32)
    params = {key: value + 0.02 * torch.randn_like(value) for key, value in params.items()}

    def evaluate(model):
        inputs = {key: value.detach().clone().requires_grad_() for key, value in params.items()}
        output = model.forward_skeleton(**inputs)
        gradients = torch.autograd.grad(output.square().mean(), tuple(inputs.values()))
        return output, gradients

    expected_output, expected_gradients = evaluate(reference)
    actual_output, actual_gradients = evaluate(accelerated)
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-4, atol=2e-4)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)


@pytest.mark.slow
def test_soma_warp_forward_and_gradients_match_torch() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    if not torch.cuda.is_available():
        pytest.skip("SOMA's Warp skinning backend requires CUDA")

    from body_models.soma.torch import SOMA

    torch.manual_seed(7)
    models = {kernel_backend: SOMA(kernel_backend=kernel_backend).cuda() for kernel_backend in ("torch", "warp")}
    params = models["torch"].get_rest_pose(batch_dims=(1,))
    params = {key: value + 0.01 * torch.randn_like(value) for key, value in params.items()}
    grad_output = torch.randn(1, models["torch"].num_vertices, 3, device="cuda")
    param_keys = tuple(params)
    results = {}

    for kernel_backend, model in models.items():
        backend_params = {key: value.detach().requires_grad_(True) for key, value in params.items()}
        vertices = model.forward_vertices(**backend_params)
        grads = torch.autograd.grad(vertices, tuple(backend_params.values()), grad_output)
        results[kernel_backend] = vertices, dict(zip(param_keys, grads, strict=True))

    torch_vertices, torch_grads = results["torch"]
    warp_vertices, warp_grads = results["warp"]
    torch.testing.assert_close(warp_vertices, torch_vertices, rtol=1e-5, atol=1e-5)
    for key in torch_grads:
        torch.testing.assert_close(warp_grads[key], torch_grads[key], rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize(
    ("name", "model_class", "kwargs"),
    model_cases.MODELS,
)
@pytest.mark.slow
def test_triton_model_gradients_match_default(
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

    def forward_and_grad(model, method, forward=None):
        model_params = {key: value.detach().clone().requires_grad_() for key, value in params.items()}
        if method == "vertices":
            forward = model.forward_vertices if forward is None else forward
            output = forward(**model_params, vertex_indices=vertex_indices)
        else:
            output = model.forward_skeleton(**model_params)
        gradients = torch.autograd.grad(
            output.square().sum(),
            tuple(model_params.values()),
            allow_unused=True,
        )
        return output, gradients

    model = torch_class(**kwargs, kernel_backend="triton").cuda()
    compiled_forward = torch.compile(model.forward_vertices, backend="eager")
    for method in ("skeleton", "vertices"):
        expected_output, expected_gradients = forward_and_grad(default_model, method)
        forward = compiled_forward if method == "vertices" else None
        actual_output, actual_gradients = forward_and_grad(model, method, forward=forward)
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-4)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            if actual is None or expected is None:
                assert actual is expected
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
