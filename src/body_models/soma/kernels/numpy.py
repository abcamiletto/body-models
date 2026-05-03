"""NumPy SOMA kernels."""

from . import base


class SomaNumpyData(base.SomaData):
    pass


def prepare_data(**data):
    data["topology"] = base.SomaTopology(
        parents_full=data.pop("parents_full"),
        parents_full_index=data.pop("parents_full_index"),
        joint_children_full=data.pop("joint_children_full"),
        joint_children_indices_full=data.pop("joint_children_indices_full"),
        skinned_vertex_indices_full=data.pop("skinned_vertex_indices_full"),
        skinned_vertex_indices_full_index=data.pop("skinned_vertex_indices_full_index"),
        kinematic_fronts_full=data.pop("kinematic_fronts_full"),
    )
    data["correctives"] = base.SomaCorrectives(
        corrective_bindpose=data.pop("corrective_bindpose"),
        corrective_W1=data.pop("corrective_W1"),
        corrective_W2_rows=data.pop("corrective_W2_rows"),
        corrective_W2_cols=data.pop("corrective_W2_cols"),
        corrective_W2_values=data.pop("corrective_W2_values"),
        corrective_W2=data.pop("corrective_W2"),
    )
    return SomaNumpyData(**data)


ops = base.SomaOps()
