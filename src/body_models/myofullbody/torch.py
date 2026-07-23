"""Torch MyoFullBody model."""

from pathlib import Path

import torch.nn as nn

from body_models.runtime import TorchRuntime
from body_models.state import torch_state
from body_models.skeletons.myofullbody.model import MyoFullBodyModel


class MyoFullBody(MyoFullBodyModel, nn.Module):
    """MyoFullBody using Torch tensors."""

    skinning_backends = ("torch",)

    def __init__(self, model_path: Path | str | None = None) -> None:
        nn.Module.__init__(self)
        MyoFullBodyModel.__init__(
            self,
            model_path,
            runtime=TorchRuntime(),
            materialize=torch_state,
        )


__all__ = ["MyoFullBody"]
