"""NumPy SOMA backend."""

from . import core

fit_rigid_transform = core.fit_rigid_transform
forward_skeleton = core.forward_skeleton
forward_vertices = core.forward_vertices
prepare_identity_shape = core.prepare_identity_shape
prepare_identity_state = core.prepare_identity_state
resolve_identity_inputs = core.resolve_identity_inputs


class SomaNumpyData(core.SomaData):
    pass


def prepare_data(**data):
    return SomaNumpyData.from_kernel_data(data)
