import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_smpl_pkl(path: Path) -> dict:
    class _ChumphyPlaceholder:
        def __init__(self, *args, **kwargs):
            self._data = None

        def __setstate__(self, state):
            if isinstance(state, dict) and "x" in state:
                self._data = np.array(state["x"])
            elif isinstance(state, dict):
                for v in state.values():
                    if isinstance(v, np.ndarray):
                        self._data = v
                        break
            self._state = state

        @property
        def r(self):
            return self._data

    class NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "scipy.sparse.csc" and name == "csc_matrix":
                from scipy.sparse import csc_matrix

                return csc_matrix
            if module.startswith("chumpy"):
                return _ChumphyPlaceholder
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = NumpyUnpickler(f, encoding="latin1").load()

    result = {}
    for k, v in data.items():
        if isinstance(v, _ChumphyPlaceholder):
            result[k] = v._data if v._data is not None else v._state
        elif hasattr(v, "toarray"):
            result[k] = v.toarray()
        elif isinstance(v, np.ndarray):
            result[k] = v
        else:
            result[k] = v

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert SMPL pkl to npz")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input} does not exist")

    print(f"Converting {args.input} -> {args.output}")
    data = load_smpl_pkl(args.input)

    save_data = {k: v for k, v in data.items() if isinstance(v, (np.ndarray, int, float, str, bool))}
    np.savez(args.output, **save_data)
    print("Done")
