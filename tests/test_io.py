import pytest

import model_cases


@pytest.mark.parametrize(
    ("name", "numpy_model", "_torch_model", "_jax_model", "model_path", "kwargs"), model_cases.MODELS
)
def test_model_loads(name, numpy_model, _torch_model, _jax_model, model_path, kwargs) -> None:
    if not model_path.exists():
        pytest.skip(f"Missing model asset: {model_path}")

    numpy_model(model_path=model_path, **kwargs)
