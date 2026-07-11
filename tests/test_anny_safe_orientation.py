import torch

from body_models.bodies.anny.backends.core import _skeleton_transforms_from_heads_tails


def test_vertical_bone_orientation_has_finite_shape_gradient() -> None:
    heads = torch.zeros((1, 1, 3), requires_grad=True)
    tails = torch.tensor([[[0.0, 1.0, 0.0]]], requires_grad=True)
    rolls = torch.eye(3)[None]
    y_axis = torch.tensor([0.0, 1.0, 0.0])
    degenerate_rotation = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

    transforms = _skeleton_transforms_from_heads_tails(
        torch,
        heads,
        tails,
        rolls,
        y_axis,
        degenerate_rotation,
    )
    transforms[..., :3, :3].sum().backward()

    assert torch.isfinite(heads.grad).all()
    assert torch.isfinite(tails.grad).all()
