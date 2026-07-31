import pytest

from body_models.brainco import BrainCoHand


@pytest.mark.parametrize("hands", ["flat", "rest"])
def test_brainco_hand_presets_forward(hands: str) -> None:
    model = BrainCoHand()
    params = model.get_rest_pose(hands=hands)

    links = model.forward_links(**params)

    assert params["hand_pose"].shape == (model.num_dofs,)
    assert links.shape[-3] == len(model.link_names)
