"""Lazy model construction from the public catalog."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any

from body_models._base import ArticulatedModel
from body_models._catalog import MODEL_SPECS
from body_models._runtime import RuntimeLike


def create_model(
    model_name: str,
    *,
    runtime: RuntimeLike = "numpy",
    **kwargs: Any,
) -> ArticulatedModel:
    """
    Create a model from its public catalog name.

    Args:
        model_name: Name returned by :func:`list_models`. Names are
            case-insensitive, and underscores are treated as hyphens.
        runtime: Runtime name or configured runtime instance.
        **kwargs: Model-specific constructor options.

    Returns:
        The requested articulated model.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    name = _normalize_name(model_name)
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}") from exc
    module = import_module(spec.module)
    model_class = getattr(module, spec.class_name)
    return model_class(**(dict(spec.defaults) | kwargs), runtime=runtime)


def list_models(*, filter: str = "") -> list[str]:
    """
    List public model factory names.

    Args:
        filter: Optional shell-style pattern such as ``"smpl*"``.

    Returns:
        Sorted matching model names.
    """
    names = sorted(MODEL_SPECS)
    if not filter:
        return names
    return [name for name in names if fnmatchcase(name, filter)]


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


__all__ = ["create_model", "list_models"]
