"""Backend-specific public model classes."""

from __future__ import annotations

from inspect import Parameter, Signature, signature
from typing import Any, Literal

from body_models._base import SkinnedModel
from body_models._runtime import RuntimeName, TorchRuntime


def model_for_backend(
    model_class: type[Any],
    backend: RuntimeName,
    *,
    module: str,
) -> type[Any]:
    """Bind a model class to one array backend."""
    has_skinning_backend = backend == "torch" and issubclass(model_class, SkinnedModel)
    backend_base: Any = model_class
    if has_skinning_backend:

        class BackendModel(backend_base):
            def __init__(
                self,
                *args: Any,
                skinning_backend: Literal["torch", "warp"] = "torch",
                **kwargs: Any,
            ) -> None:
                _reject_runtime(kwargs, model_class)
                runtime = TorchRuntime(skinning_backend=skinning_backend)
                super().__init__(*args, runtime=runtime, **kwargs)

    else:

        class BackendModel(backend_base):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _reject_runtime(kwargs, model_class)
                super().__init__(*args, runtime=backend, **kwargs)

    BackendModel.__name__ = model_class.__name__
    BackendModel.__qualname__ = model_class.__qualname__
    BackendModel.__module__ = module
    BackendModel.__doc__ = model_class.__doc__
    BackendModel.__signature__ = _backend_signature(model_class, has_skinning_backend)
    return BackendModel


def _backend_signature(model_class: type[Any], has_skinning_backend: bool) -> Signature:
    model_signature = signature(model_class)
    parameters = list(model_signature.parameters.values())
    runtime_index = next(index for index, parameter in enumerate(parameters) if parameter.name == "runtime")
    parameters.pop(runtime_index)
    if has_skinning_backend:
        parameters.insert(
            runtime_index,
            Parameter(
                "skinning_backend",
                kind=Parameter.KEYWORD_ONLY,
                default="torch",
                annotation=Literal["torch", "warp"],
            ),
        )
    return model_signature.replace(parameters=parameters)


def _reject_runtime(kwargs: dict[str, Any], model_class: type[Any]) -> None:
    if "runtime" in kwargs:
        raise TypeError(f"{model_class.__name__}() got an unexpected keyword argument 'runtime'")


__all__ = ["model_for_backend"]
