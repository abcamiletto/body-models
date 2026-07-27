"""PyTorch lifecycle adapter for backend-neutral model objects."""

from typing import Any

from torch import nn


class TorchModule(nn.Module):
    """Expose a Torch-backed model through ``nn.Module`` state management."""

    def __init__(self, model: Any) -> None:
        if model.runtime.backend != "torch":
            raise TypeError("as_module() requires a Torch-backed model.")
        super().__init__()
        for name in model._state_fields:
            value = getattr(model, name)
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
