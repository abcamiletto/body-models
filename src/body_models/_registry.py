"""Lazy model construction from the public catalog."""

from __future__ import annotations

from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any

from body_models._base import SkinnedModel
from body_models._catalog import MODEL_SPECS
from body_models._runtime import RuntimeName


def create_model(
    model_name: str,
    *,
    runtime: RuntimeName = "numpy",
    **kwargs: Any,
) -> SkinnedModel:
    """
    Create a model from its public catalog name.

    Args:
        model_name: Name returned by :func:`list_models`. Names are
            case-insensitive, and underscores are treated as hyphens.
        runtime: Array backend name.
        **kwargs: Model-specific constructor options.

    Returns:
        The requested articulated model.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    name = _normalize_name(model_name)
    if runtime not in ("numpy", "torch", "jax"):
        raise ValueError(f"Unknown runtime {runtime!r}. Expected numpy, torch, or jax.")
    try:
        spec = MODEL_SPECS[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}") from exc
    module = import_module(f"{spec.module}.{runtime}")
    model_class = getattr(module, spec.class_name)
    return model_class(**(dict(spec.defaults) | kwargs))


def list_models(*, pattern: str | None = None) -> list[str]:
    """
    List public model factory names.

    Args:
        pattern: Optional case-insensitive shell-style pattern such as ``"smpl*"``.

    Returns:
        Sorted matching model names.
    """
    names = sorted(MODEL_SPECS)
    if pattern is None:
        return names
    pattern = _normalize_name(pattern)
    return [name for name in names if fnmatchcase(name, pattern)]


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


__all__ = ["create_model", "list_models"]
