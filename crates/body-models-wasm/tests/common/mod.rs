//! Shared test utilities for CPU and GPU integration tests.

use std::path::PathBuf;

use body_models_wasm::SmplModel;

pub const NUM_CASES: usize = 5;
pub const RTOL: f32 = 1e-4;
pub const ATOL: f32 = 1e-4;
pub const TEST_SHAPE_DIM: usize = 10;

pub fn model_npz_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/assets/smpl/model/SMPL_NEUTRAL.npz")
}

pub fn model_bin_path() -> PathBuf {
    PathBuf::from("/tmp/smpl_neutral.bin")
}

pub fn assert_allclose(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        name,
        actual.len(),
        expected.len()
    );

    for i in 0..actual.len() {
        let a = actual[i];
        let e = expected[i];
        let abs_diff = (a - e).abs();
        let tol = ATOL + RTOL * e.abs();
        assert!(
            abs_diff <= tol,
            "{}: mismatch at index {}: actual={}, expected={}, abs_diff={}, tol={}",
            name,
            i,
            a,
            e,
            abs_diff,
            tol
        );
    }
}

pub fn load_model() -> Option<SmplModel> {
    let npz_path = model_npz_path();
    if npz_path.exists() {
        let bytes = std::fs::read(&npz_path).unwrap();
        return Some(SmplModel::from_bytes(&bytes).unwrap());
    }
    let bin_path = model_bin_path();
    if bin_path.exists() {
        let bytes = std::fs::read(&bin_path).unwrap();
        return Some(SmplModel::from_bytes(&bytes).unwrap());
    }
    None
}

/// Per-case input data.
pub struct CaseData {
    pub shape: Vec<f32>,
    pub body_pose: Vec<f32>,
    pub pelvis_rotation: Vec<f32>,
    pub global_translation: Vec<f32>,
}

impl CaseData {
    pub fn pelvis_opt(&self) -> Option<&[f32]> {
        Some(&self.pelvis_rotation)
    }
    pub fn translation_opt(&self) -> Option<&[f32]> {
        Some(&self.global_translation)
    }
}

pub fn load_case(idx: usize, body_pose_len: usize) -> CaseData {
    let offset = idx as f32 * 0.7;
    let shape: Vec<f32> = (0..TEST_SHAPE_DIM)
        .map(|s| ((s as f32 + offset) * 0.31415).sin())
        .collect();
    let body_pose: Vec<f32> = (0..body_pose_len)
        .map(|p| ((p as f32 + offset) * 0.1234).sin() * 0.3)
        .collect();
    let pelvis_rotation = vec![0.1 + offset * 0.01, -0.2 + offset * 0.01, 0.15];
    let global_translation = vec![offset * 0.01, -offset * 0.02, offset * 0.03];
    CaseData {
        shape,
        body_pose,
        pelvis_rotation,
        global_translation,
    }
}

/// Concatenated inputs for all synthetic cases.
pub struct BatchedInputs {
    pub shape: Vec<f32>,
    pub body_pose: Vec<f32>,
    pub pelvis: Vec<f32>,
    pub translation: Vec<f32>,
}

pub fn load_batched_inputs(body_pose_len: usize) -> BatchedInputs {
    let mut out = BatchedInputs {
        shape: Vec::new(),
        body_pose: Vec::new(),
        pelvis: Vec::new(),
        translation: Vec::new(),
    };
    for idx in 0..NUM_CASES {
        let c = load_case(idx, body_pose_len);
        out.shape.extend_from_slice(&c.shape);
        out.body_pose.extend_from_slice(&c.body_pose);
        out.pelvis.extend_from_slice(&c.pelvis_rotation);
        out.translation.extend_from_slice(&c.global_translation);
    }
    out
}
