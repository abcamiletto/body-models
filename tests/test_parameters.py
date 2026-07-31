"""Public model-parameter contracts."""

import model_cases
import numpy as np
import pytest

from body_models import Joint, ParameterSpec


@pytest.mark.parametrize(("_name", "model_class", "kwargs"), model_cases.MODELS)
def test_model_contract(_name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    joint_names = model.joint_names
    parents = model.parents

    assert len(parents) == len(joint_names) == model.num_joints
    assert parents.count(-1) == 1
    assert all(parent == -1 or 0 <= parent < model.num_joints for parent in parents)
    assert all(isinstance(joint, Joint) for joint in model.common_joints)
    for joint, native_name in model.common_joints.items():
        assert native_name in joint_names
        assert model.joint_index(joint) == joint_names.index(native_name)

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
