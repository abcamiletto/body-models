"""Torch MyoFullBody model."""

from pathlib import Path

from torch import nn

from body_models._runtime import TorchRuntime
from body_models.myofullbody._model import MyoFullBodyModel


class MyoFullBody(MyoFullBodyModel, nn.Module):
    """MyoFullBody using Torch tensors."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        nn.Module.__init__(self)
        MyoFullBodyModel.__init__(
            self,
            model_path,
            runtime=TorchRuntime(),
        )


__all__ = ["MyoFullBody"]
