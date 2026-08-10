"""Backend-specific public model classes."""

from __future__ import annotations

from inspect import Parameter, Signature, signature
from typing import Any

from body_models._base import SkinnedModel
from body_models._runtime import ArrayRuntime, JaxRuntime, KernelBackend, NumpyRuntime, RuntimeName, TorchRuntime

_RUNTIME_CLASSES: dict[RuntimeName, type[ArrayRuntime]] = {
    "numpy": NumpyRuntime,
    "torch": TorchRuntime,
    "jax": JaxRuntime,
}


def model_for_backend(
    model_class: type[Any],
    backend: RuntimeName,
    *,
    module: str,
) -> type[Any]:
    """Bind a model class to one array backend."""
    runtime_class = _RUNTIME_CLASSES[backend]
    has_kernel_backend = backend == "torch" and issubclass(model_class, SkinnedModel)
    backend_base: Any = model_class
    if backend == "torch":
        from torch import nn

        torch_base: Any = nn.Module

    if has_kernel_backend:

        class BackendModel(backend_base, torch_base):
            def __init__(
                self,
                *args: Any,
                kernel_backend: KernelBackend = "torch",
                **kwargs: Any,
            ) -> None:
                _reject_runtime(kwargs, model_class)
                nn.Module.__init__(self)
                runtime = TorchRuntime(kernel_backend=kernel_backend)
                super().__init__(*args, runtime=runtime, **kwargs)

    elif backend == "torch":

        class BackendModel(backend_base, torch_base):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _reject_runtime(kwargs, model_class)
                nn.Module.__init__(self)
                super().__init__(*args, runtime=runtime_class(), **kwargs)

    else:

        class BackendModel(backend_base):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _reject_runtime(kwargs, model_class)
                super().__init__(*args, runtime=runtime_class(), **kwargs)

    BackendModel.__name__ = model_class.__name__
    BackendModel.__qualname__ = model_class.__qualname__
    BackendModel.__module__ = module
    BackendModel.__doc__ = model_class.__doc__
    backend_signature = _backend_signature(model_class, has_kernel_backend)
    BackendModel.__signature__ = backend_signature
    backend_init: Any = BackendModel.__init__
    backend_init.__module__ = module
    backend_init.__qualname__ = f"{model_class.__qualname__}.__init__"
    self_parameter = Parameter("self", kind=Parameter.POSITIONAL_ONLY)
    initializer_signature = backend_signature.replace(
        parameters=(self_parameter, *backend_signature.parameters.values())
    )
    backend_init.__dict__["__signature__"] = initializer_signature
    return BackendModel


def _backend_signature(model_class: type[Any], has_kernel_backend: bool) -> Signature:
    model_signature = signature(model_class)
    parameters = list(model_signature.parameters.values())
    runtime_index = next(index for index, parameter in enumerate(parameters) if parameter.name == "runtime")
    parameters.pop(runtime_index)
    if has_kernel_backend:
        kernel_backend = Parameter(
            "kernel_backend",
            kind=Parameter.KEYWORD_ONLY,
            default="torch",
            annotation=KernelBackend,
        )
        parameters.insert(runtime_index, kernel_backend)
    return model_signature.replace(parameters=parameters)


def _reject_runtime(kwargs: dict[str, Any], model_class: type[Any]) -> None:
    if "runtime" in kwargs:
        raise TypeError(f"{model_class.__name__}() got an unexpected keyword argument 'runtime'")


__all__ = ["model_for_backend"]
