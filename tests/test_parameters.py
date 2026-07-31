"""Public model-parameter contracts."""

import model_cases
import numpy as np
import pytest

from body_models import ParameterSpec


@pytest.mark.parametrize(("_name", "model_class", "kwargs"), model_cases.MODELS)
def test_parameter_spec_describes_rest_parameters(_name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    batch_shape = (2, 3)
    parameters = model.get_rest_pose(batch_dims=batch_shape, dtype=np.float32)

    assert parameters.keys() == model.parameter_spec.keys()
    roles = [spec.role for spec in model.parameter_spec.values()]
    assert roles == sorted(roles, key=("identity", "pose", "transform").index)
    for name, spec in model.parameter_spec.items():
        assert isinstance(spec, ParameterSpec)
        assert parameters[name].shape == (*batch_shape, *spec.shape)
        assert parameters[name].dtype == np.float32


def test_parameter_spec_exposes_rotation_representation() -> None:
    from body_models.smpl import SMPL

    model = SMPL(gender="neutral", rotation_type="rotmat")

    assert model.parameter_spec["shape"] == ParameterSpec((10,), "identity")
    assert model.parameter_spec["body_pose"] == ParameterSpec.rotation("rotmat", count=23)
    assert model.parameter_spec["global_rotation"] == ParameterSpec.rotation(
        "rotmat",
        role="transform",
    )
