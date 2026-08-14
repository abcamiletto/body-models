"""Left/right joint pairing across every model."""

import model_cases
import pytest

from body_models import Joint
from body_models.mano.numpy import MANO
from body_models.smpl.numpy import SMPL

pytestmark = pytest.mark.fast

# Pair counts per model, so an asset change that drops a pair fails loudly.
EXPECTED_PAIR_COUNTS = {
    "anny": 70,
    "flame": 1,
    "garment_measurements": 26,
    "mano": 0,
    "mhr": 54,
    "skel": 10,
    "smpl": 9,
    "smplh": 23,
    "smplx": 24,
    "soma": 34,
}


def stem(name, affix):
    """Joint name without its side affix."""
    assert name.startswith(affix) or name.endswith(affix), f"{name!r} is not sided by {affix!r}"
    return name[len(affix) :] if name.startswith(affix) else name[: -len(affix)]


def canonical_pairs(model):
    """Canonical left/right joints both mapped by this model."""
    for member, left in Joint.__members__.items():
        if not member.startswith("LEFT_"):
            continue
        right = Joint.__members__.get(member.replace("LEFT", "RIGHT"))
        if right is not None and left in model.common_joints and right in model.common_joints:
            yield left, right


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_pair_count_matches_snapshot(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)

    assert len(model.symmetric_joints) == EXPECTED_PAIR_COUNTS[name]


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_every_joint_is_paired_once_or_midline(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)

    paired = [index for pair in model.symmetric_joints for index in pair]

    assert len(paired) == len(set(paired)), "a joint appears in more than one pair"
    assert all(0 <= index < model.num_joints for index in paired)
    assert len(model.symmetric_joints) * 2 + len(set(range(model.num_joints)) - set(paired)) == model.num_joints


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_pairs_are_named_for_opposite_sides(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)
    names = model.joint_names
    left_affix, right_affix = model._SIDE_AFFIXES or ("", "")

    for left, right in model.symmetric_joints:
        assert left != right
        assert stem(names[left], left_affix) == stem(names[right], right_affix)


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_pairs_cover_every_canonical_left_right_joint(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)

    for left, right in canonical_pairs(model):
        pair = (model.joint_index(left), model.joint_index(right))
        assert pair in model.symmetric_joints, f"{left.value}/{right.value} missing"


@pytest.mark.parametrize(("name", "model_class", "kwargs"), model_cases.MODELS)
def test_documented_swap_order_is_an_involution(name, model_class, kwargs) -> None:
    model = model_class(**kwargs)

    order = list(range(model.num_joints))
    for left, right in model.symmetric_joints:
        order[left], order[right] = right, left

    assert sorted(order) == list(range(model.num_joints))
    assert [order[index] for index in order] == list(range(model.num_joints))


def test_one_sided_skeleton_has_no_pairs() -> None:
    assert MANO(side="left").symmetric_joints == ()
    assert MANO(side="right").symmetric_joints == ()


def test_unpaired_sided_joint_is_rejected() -> None:
    class Lopsided(SMPL):
        @property
        def joint_names(self) -> list[str]:
            return ["pelvis", "left_hip"]

    with pytest.raises(ValueError, match="left_hip"):
        _ = Lopsided(gender="neutral").symmetric_joints
