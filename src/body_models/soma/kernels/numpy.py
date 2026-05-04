"""NumPy SOMA kernels."""

from . import base

fit_rigid_transform = base.fit_rigid_transform
prepare_identity_shape = base.prepare_identity_shape
resolve_identity_inputs = base.resolve_identity_inputs


class SomaNumpyData(base.SomaData):
    pass


def prepare_data(**data):
    return SomaNumpyData.from_kernel_data(data)
