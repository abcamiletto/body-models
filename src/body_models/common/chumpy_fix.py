"""Load model files that may contain legacy chumpy objects."""

from __future__ import annotations

import pickle
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

PathLike = Path | str

__all__ = ["load_model_dict"]


def load_model_dict(model_path: PathLike) -> dict[str, Any]:
    model_path = Path(model_path)
    if model_path.suffix == ".npz":
        with _suppress_numpy_align_warning():
            with np.load(model_path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}
    if model_path.suffix == ".pkl":
        return _load_pickle_dict(model_path)
    raise ValueError(f"Expected a .pkl or .npz file, got: {model_path}")


def _load_pickle_dict(model_path: Path) -> dict[str, Any]:
    with open(model_path, "rb") as f:
        with _suppress_numpy_align_warning():
            data = _CompatUnpickler(f, encoding="latin1").load()
    return {key: _array_value(value) for key, value in data.items()}


@contextmanager
def _suppress_numpy_align_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=np.exceptions.VisibleDeprecationWarning,
        )
        yield


def _array_value(value: Any) -> Any:
    if isinstance(value, _ChumpyPlaceholder):
        return value.data
    if sparse.issparse(value):
        return value.toarray()
    return value


class _ChumpyPlaceholder:
    def __setstate__(self, state):
        if isinstance(state, dict) and "x" in state:
            self.data = np.asarray(state["x"])
            return
        if isinstance(state, dict) and "a" in state and "idxs" in state:
            source = state["a"]
            source_data = source.data if isinstance(source, _ChumpyPlaceholder) else np.asarray(source)
            data = np.asarray(source_data).reshape(-1)[np.asarray(state["idxs"], dtype=np.int64)]
            if "preferred_shape" in state:
                data = data.reshape(tuple(state["preferred_shape"]))
            self.data = data
            return
        if isinstance(state, dict):
            for value in state.values():
                if isinstance(value, np.ndarray):
                    self.data = value
                    return
        self.data = state


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "scipy.sparse.csc" and name == "csc_matrix":
            from scipy.sparse import csc_matrix

            return csc_matrix
        if module.startswith("chumpy"):
            return _ChumpyPlaceholder
        return super().find_class(module, name)
