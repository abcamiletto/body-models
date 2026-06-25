import numpy as np
import pytest

import model_cases
from body_models.bodies.soma.io import validate_path


@pytest.mark.parametrize(("name", "numpy_model", "_torch_model", "_jax_model", "kwargs"), model_cases.MODELS)
def test_model_loads(name, numpy_model, _torch_model, _jax_model, kwargs) -> None:
    numpy_model(**kwargs)


def test_soma_slim_npz_asset_layout_requires_sidecars(tmp_path) -> None:
    np.savez(tmp_path / "SOMA_neutral.npz", mean=np.zeros((1, 3), dtype=np.float32))
    (tmp_path / "correctives_model.pt").touch()

    with pytest.raises(FileNotFoundError, match="Missing sidecars"):
        validate_path(tmp_path)
