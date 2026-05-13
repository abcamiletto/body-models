import numpy as np
import pytest

import model_cases


@pytest.mark.parametrize(
    ("name", "_numpy_model", "torch_model", "jax_model", "model_path", "kwargs"), model_cases.MODELS
)
def test_torch_and_jax_gradients_match_finite_difference(
    name, _numpy_model, torch_model, jax_model, model_path, kwargs
) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    torch = pytest.importorskip("torch")
    torch_instance = torch_model(model_path=model_path, **kwargs)
    torch_instance.double()
    torch_params = torch_instance.get_rest_pose(batch_size=1, dtype=torch.float64)
    for torch_key in gradient_keys(torch_params):
        torch_value = torch_params[torch_key].clone().requires_grad_(True)
        torch_params[torch_key] = torch_value
        torch_loss_value = torch_instance.forward_vertices(**torch_params)[..., :8, :].sum()
        if torch_loss_value.requires_grad:
            break
        torch_params[torch_key] = torch_value.detach()
    torch_loss_value.backward()

    torch_auto = torch_value.grad.reshape(-1)[0].item()
    torch_numeric = finite_difference(
        lambda value: torch_loss(torch_instance, torch_params, torch_key, value), torch_value
    )
    np.testing.assert_allclose(torch_auto, torch_numeric, rtol=1e-2, atol=1e-2)

    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    jax_instance = jax_model(model_path=model_path, **kwargs)
    jax_params = jax_instance.get_rest_pose(batch_size=1, dtype=jnp.float64)
    jax_key = torch_key
    jax_value = jax_params[jax_key]

    def jax_loss(value):
        params = jax_params.copy()
        params[jax_key] = value
        return jax_instance.forward_vertices(**params)[..., :8, :].sum()

    jax_auto = np.asarray(jax.grad(jax_loss)(jax_value)).reshape(-1)[0]
    jax_numeric = finite_difference(lambda value: float(jax_loss(jnp.asarray(value))), jax_value)
    np.testing.assert_allclose(jax_auto, jax_numeric, rtol=1e-2, atol=1e-2)


def torch_loss(model, params, key, value) -> float:
    import torch

    params = params.copy()
    params[key] = torch.as_tensor(value, dtype=params[key].dtype)
    with torch.no_grad():
        return model.forward_vertices(**params)[..., :8, :].sum().item()


def gradient_keys(params: dict) -> list[str]:
    keys = [key for key, value in params.items() if np.asarray(value).size]
    if "global_translation" in keys:
        keys.remove("global_translation")
        keys.insert(0, "global_translation")
    return keys


def finite_difference(loss, value, eps: float = 1e-4) -> float:
    value = value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)
    plus = value.copy()
    minus = value.copy()
    plus.reshape(-1)[0] += eps
    minus.reshape(-1)[0] -= eps
    return (loss(plus) - loss(minus)) / (2 * eps)
