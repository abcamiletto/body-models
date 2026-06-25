import numpy as np
import pytest

import model_assets
from body_models.bodies.soma.numpy import SOMA
from body_models.bodies.soma.pose import unpack_pose


def test_soma_021_matches_upstream_pure_lbs() -> None:
    torch = pytest.importorskip("torch")
    upstream_soma = pytest.importorskip("soma")

    model_path = model_assets.get_model_file("soma")
    required = [
        model_path / "SOMA_neutral.npz",
        model_path / "SOMA_template_rig.usda",
        model_path / "SOMA_procedural_transforms.json",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("SOMA 0.2.1 assets are not available")

    upstream = upstream_soma.SOMALayer(
        data_root=model_path,
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        lod="mid",
        correctives_model_path=None,
    )
    model = SOMA(model_path=model_path, model_type="soma", rotation_type="axis_angle")

    shape = np.zeros((1, 128), dtype=np.float32)
    poses = [
        np.zeros((1, 77, 3), dtype=np.float32),
        np.zeros((1, 77, 3), dtype=np.float32),
        np.zeros((1, 77, 3), dtype=np.float32),
    ]
    poses[1][0, 0, 0] = 0.4
    poses[2][0, 38, 0] = 0.8

    for pose in poses:
        with torch.no_grad():
            upstream_vertices = upstream(
                poses=torch.as_tensor(pose),
                identity_coeffs=torch.zeros(1, 128),
                apply_correctives=False,
            )["vertices"].detach().numpy()

        global_rotation, body_pose, head_pose, hand_pose = unpack_pose(np, pose)
        identity = model.prepare_identity(shape)
        prepared_pose = model.prepare_pose(body_pose, head_pose, hand_pose, global_rotation, identity=identity)
        vertices = model._kernel.forward_vertices(
            data=model.weights,
            global_translation=None,
            vertex_indices=None,
            rotation_type=model.rotation_type,
            rest_vertices=identity["rest_vertices"],
            skinning_transforms=prepared_pose["skinning_transforms"],
            pose_offsets=np.zeros_like(prepared_pose["pose_offsets"]),
            xp=np,
        )

        np.testing.assert_allclose(vertices, upstream_vertices, rtol=2e-3, atol=2e-3)
