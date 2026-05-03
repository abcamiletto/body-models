"""NumPy SOMA kernels."""

from . import base


class SomaNumpyData(base.SomaData):
    pass


def prepare_data(**data):
    return SomaNumpyData.from_kernel_data(data)


ops = base.SomaOps()
