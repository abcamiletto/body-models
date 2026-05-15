import numpy as np
import pytest

import model_cases


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
        torch_loss_value = torch_instance.forward_vertices(**torch_params)[..., :8, :].sum()
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
        torch_plus_loss = torch_instance.forward_vertices(**plus_params)[..., :8, :].sum().item()
        torch_minus_loss = torch_instance.forward_vertices(**minus_params)[..., :8, :].sum().item()
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
        return jax_instance.forward_vertices(**params)[..., :8, :].sum()

    jax_auto = np.asarray(jax.grad(jax_loss)(jax_value)).reshape(-1)[0]
    jax_plus = np.asarray(jax_value).copy()
    jax_minus = np.asarray(jax_value).copy()
    jax_plus.reshape(-1)[0] += 1e-4
    jax_minus.reshape(-1)[0] -= 1e-4
    jax_numeric = (float(jax_loss(jnp.asarray(jax_plus))) - float(jax_loss(jnp.asarray(jax_minus)))) / 2e-4
    np.testing.assert_allclose(jax_auto, jax_numeric, rtol=1e-2, atol=1e-2)
