"""PyTorch lifecycle adapter for backend-neutral model objects."""

from typing import Any

from torch import nn

from body_models import _base


class TorchModule(nn.Module):
    """Cached ``nn.Module`` view over a Torch-backed model's shared state."""

    def __init__(self, model: Any) -> None:
        if model.runtime.backend != "torch":
            raise TypeError("as_module() requires a Torch-backed model.")
        super().__init__()
        for name in model._state_fields:
            value = getattr(model, name)
            if isinstance(value, _base._ArticulatedModel):
                value = value.as_module()
            if value is not None:
                self.add_module(name, value)
        object.__setattr__(self, "_wrapped_model", model)

    @property
    def model(self) -> Any:
        """The backend-neutral model wrapped by this module."""
        return object.__getattribute__(self, "_wrapped_model")

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = self.__dict__.get("_wrapped_model")
            if model is None:
                raise
            return getattr(model, name)


__all__ = ["TorchModule"]
