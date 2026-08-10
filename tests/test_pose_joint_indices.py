"""Pose parameter metadata aligned with the public skeleton."""

import model_cases
import numpy as np
import pytest

from body_models.anny import _pose as anny_pose
from body_models.garment_measurements import _pose as garment_pose
from body_models.skel import _pose as skel_pose
from body_models.smplx import SMPLX
from body_models.soma import _pose as soma_pose


@pytest.mark.fast
@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_pose_joint_indices_select_the_canonical_skeleton(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    params = model.get_rest_pose()
    skeleton = model.forward_skeleton(**params)
    pose_parameters = {parameter for parameter, spec in model.parameter_spec.items() if spec.role == "pose"}

    assert set(model.pose_joint_indices) == pose_parameters
    for joint_indices in model.pose_joint_indices.values():
        assert joint_indices == tuple(sorted(set(joint_indices)))
        assert all(0 <= joint < model.num_joints for joint in joint_indices)
        selected = model.forward_skeleton(**params, joint_indices=joint_indices)
        np.testing.assert_array_equal(selected, skeleton[..., list(joint_indices), :, :])


@pytest.mark.fast
def test_pose_layouts_describe_interleaved_joint_orders() -> None:
    assert SMPLX._POSE_LAYOUT.joint_indices == {
        "pelvis_rotation": (0,),
        "body_pose": tuple(range(1, 22)),
        "head_pose": tuple(range(22, 25)),
        "hand_pose": tuple(range(25, 55)),
    }
    assert anny_pose.POSE_LAYOUT.joint_indices == {
        "root_rotation": (0,),
        "body_pose": (*range(1, 55), *range(74, 81), *range(100, 103)),
        "hand_pose": (*range(55, 74), *range(81, 100)),
        "head_pose": tuple(range(103, 163)),
    }
    assert soma_pose.POSE_LAYOUT.joint_indices == {
        "root_rotation": (0,),
        "body_pose": (*range(1, 6), *range(11, 15), *range(39, 43), *range(67, 77)),
        "head_pose": tuple(range(6, 11)),
        "hand_pose": (*range(15, 39), *range(43, 67)),
    }
    assert garment_pose.POSE_LAYOUT.joint_indices == {
        "pelvis_rotation": (0,),
        "body_pose": (*range(1, 6), *range(9, 15), *range(30, 36), *range(51, 59)),
        "head_pose": tuple(range(6, 9)),
        "hand_pose": (*range(15, 30), *range(36, 51)),
    }
    assert skel_pose.POSE_LAYOUT.joint_indices == {
        "body_pose": (*range(13), *range(14, 24)),
        "head_pose": (13,),
    }
