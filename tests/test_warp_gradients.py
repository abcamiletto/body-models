"""Gradient tests for shared Warp body-model kernels."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("warp")

from body_models.bodies.garment_measurements.backends import core as garment_core  # noqa: E402
from body_models.bodies.smpl.backends import warp as smpl_warp  # noqa: E402
from body_models.garment_measurements.torch import GarmentMeasurements  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_affine_skinning_gradients_match_torch() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    batch_size, num_vertices, num_joints = 2, 5, 4

    vertices = torch.randn(batch_size, num_vertices, 3, device=device, generator=generator, requires_grad=True)
    transforms = torch.randn(
        batch_size,
        num_joints,
        4,
        4,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    weights = torch.rand(num_vertices, num_joints, device=device, generator=generator)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    joint_indices = torch.arange(num_joints, device=device).expand(num_vertices, num_joints)
    grad_output = torch.randn(batch_size, num_vertices, 3, device=device, generator=generator)

    actual = smpl_warp.warp_affine_blend_skinning(vertices, transforms, joint_indices, weights)
    actual.backward(grad_output)
    actual_grads = vertices.grad, transforms.grad

    ref_vertices = vertices.detach().requires_grad_(True)
    ref_transforms = transforms.detach().requires_grad_(True)
    expected = garment_core.linear_blend_skinning(torch, ref_vertices, ref_transforms, weights)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_grads[0], ref_vertices.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_grads[1], ref_transforms.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warp_forward_kinematics_gradients_match_torch() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1)
    batch_size, num_joints = 2, 4
    parents = torch.tensor([-1, 0, 1, 1], device=device)
    rotations = torch.randn(
        batch_size,
        num_joints,
        3,
        3,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    translations = torch.randn(
        batch_size,
        num_joints,
        3,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    grad_output = torch.randn(batch_size, num_joints, 4, 4, device=device, generator=generator)

    actual = smpl_warp.warp_forward_kinematics(rotations, translations, parents)
    actual.backward(grad_output)
    actual_grads = rotations.grad, translations.grad

    ref_rotations = rotations.detach().requires_grad_(True)
    ref_translations = translations.detach().requires_grad_(True)
    world = []
    for joint, parent in enumerate(parents.tolist()):
        upper = torch.concat([ref_rotations[:, joint], ref_translations[:, joint, :, None]], dim=-1)
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(batch_size, 1, 4)
        local = torch.concat([upper, bottom], dim=-2)
        world.append(local if parent < 0 else world[parent] @ local)
    expected = torch.stack(world, dim=1)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_grads[0], ref_rotations.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_grads[1], ref_translations.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_garment_warp_gradients_match_torch() -> None:
    device = torch.device("cuda")
    torch_model = GarmentMeasurements().to(device)
    warp_model = GarmentMeasurements(kernel="warp").to(device)
    params = torch_model.get_rest_pose(batch_dims=(2,))
    generator = torch.Generator(device=device).manual_seed(2)
    for key, value in params.items():
        params[key] = (value + 0.1 * torch.randn(value.shape, device=device, generator=generator)).requires_grad_()

    def forward_and_grad(model):
        model_params = {key: value.detach().clone().requires_grad_() for key, value in params.items()}
        vertices = model.forward_vertices(**model_params)
        gradients = torch.autograd.grad(vertices.square().sum(), tuple(model_params.values()))
        return vertices, gradients

    expected_vertices, expected_gradients = forward_and_grad(torch_model)
    actual_vertices, actual_gradients = forward_and_grad(warp_model)

    torch.testing.assert_close(actual_vertices, expected_vertices, rtol=1e-4, atol=1e-4)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
