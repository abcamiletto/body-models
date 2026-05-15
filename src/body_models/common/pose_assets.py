from importlib.resources import files

import numpy as np


def load_npz(package: str, filename: str = "poses.npz") -> dict[str, np.ndarray]:
    path = files(package) / "assets" / filename
    with path.open("rb") as file:
        data = np.load(file)
        return {key: _unpack(data[key]) for key in data.files}


def _unpack(value: np.ndarray):
    if value.dtype.names is None:
        return value
    return {name: value[name] for name in value.dtype.names}
