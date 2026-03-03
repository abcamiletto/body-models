//! SMPL-X CPU smoke tests.
//! Skips gracefully when local SMPL-X asset is unavailable.

use std::path::PathBuf;

use body_models_wasm::SmplxModel;

fn smplx_npz_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/assets/smplx/model/SMPLX_NEUTRAL.npz")
}

#[test]
fn test_smplx_forward_smoke() {
    let path = smplx_npz_path();
    if !path.exists() {
        eprintln!("Skipping SMPL-X smoke test: {} not found", path.display());
        return;
    }

    let bytes = std::fs::read(&path).unwrap();
    let model = SmplxModel::from_npz(&bytes).unwrap();

    let b = 1usize;
    let shape = vec![0.0f32; b * 10];
    let body_pose = vec![0.0f32; b * 21 * 3];
    let hand_pose = vec![0.0f32; b * 30 * 3];
    let head_pose = vec![0.0f32; b * 3 * 3];
    let pelvis = vec![0.0f32; b * 3];
    let translation = vec![0.0f32; b * 3];

    let verts = model.forward_vertices(
        &shape,
        &body_pose,
        &hand_pose,
        &head_pose,
        None,
        Some(&pelvis),
        None,
        Some(&translation),
        false,
    );
    assert_eq!(verts.len(), b * model.data.num_vertices * 3);

    let skel = model.forward_skeleton(
        &shape,
        &body_pose,
        &hand_pose,
        &head_pose,
        None,
        Some(&pelvis),
        None,
        Some(&translation),
        false,
    );
    assert_eq!(skel.len(), b * model.data.num_joints * 16);
}

