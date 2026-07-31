"""Lazy model construction from the public catalog."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any

from body_models._base import ArticulatedModel
from body_models._catalog import MODEL_SPECS, ModelSpec
from body_models._runtime import RuntimeLike


def create_model(
    model_name: str,
    *,
    runtime: RuntimeLike = "numpy",
    **kwargs: Any,
) -> ArticulatedModel:
    """Create a model by catalog name and array runtime."""
    name = _normalize_name(model_name)
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}") from exc
    module = import_module(spec.module)
    model_class = getattr(module, spec.class_name)
    return model_class(**(dict(spec.defaults) | kwargs), runtime=runtime)


def get_model_spec(model_name: str) -> ModelSpec:
    """Return immutable catalog metadata for a model name."""
    name = _normalize_name(model_name)
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}") from exc


def list_models(*, filter: str = "") -> list[str]:
    """List catalog names, optionally filtered with a shell-style pattern."""
    names = sorted(MODEL_SPECS)
    if not filter:
        return names
    return [name for name in names if fnmatchcase(name, filter)]


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


__all__ = ["create_model", "get_model_spec", "list_models"]
