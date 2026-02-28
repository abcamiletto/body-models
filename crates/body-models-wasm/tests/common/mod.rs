//! Shared test utilities for CPU and GPU integration tests.

use std::path::PathBuf;

use body_models_wasm::SmplModel;

pub const NUM_CASES: usize = 5;
pub const RTOL: f32 = 1e-4;
pub const ATOL: f32 = 1e-4;

pub fn reference_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference_data")
}

pub fn model_npz_path() -> PathBuf {
    PathBuf::from("/CT/aboscolo/work/experiments/data-samples/smpl/SMPL_NEUTRAL.npz")
}

pub fn model_bin_path() -> PathBuf {
    PathBuf::from("/tmp/smpl_neutral.bin")
}

pub fn load_f32_bin(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!("Failed to read {}: {}", path.display(), e)
    });
    assert_eq!(bytes.len() % 4, 0, "File size not multiple of 4: {}", path.display());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub fn assert_allclose(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        name, actual.len(), expected.len()
    );

    let mut max_abs_diff = 0.0f32;
    let mut worst_idx = 0;

    for i in 0..actual.len() {
        let a = actual[i];
        let e = expected[i];
        let abs_diff = (a - e).abs();

        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            worst_idx = i;
        }

        let tol = ATOL + RTOL * e.abs();
        assert!(
            abs_diff <= tol,
            "{}: mismatch at index {}: actual={}, expected={}, abs_diff={}, tol={}",
            name, i, a, e, abs_diff, tol
        );
    }

    eprintln!("  {}: max_abs={:.2e} (idx {}) ✓", name, max_abs_diff, worst_idx);
}

pub fn load_model() -> SmplModel {
    let npz_path = model_npz_path();
    if npz_path.exists() {
        let bytes = std::fs::read(&npz_path).unwrap();
        return SmplModel::from_bytes(&bytes).unwrap();
    }
    let bin_path = model_bin_path();
    if bin_path.exists() {
        let bytes = std::fs::read(&bin_path).unwrap();
        return SmplModel::from_bytes(&bytes).unwrap();
    }
    panic!("SMPL model not found at {} or {}", npz_path.display(), bin_path.display());
}

/// Per-case input data loaded from reference files.
pub struct CaseData {
    pub shape: Vec<f32>,
    pub body_pose: Vec<f32>,
    pub pelvis_rotation: Vec<f32>,
    pub global_translation: Vec<f32>,
}

impl CaseData {
    pub fn pelvis_opt(&self) -> Option<&[f32]> {
        if self.pelvis_rotation.iter().any(|&v| v != 0.0) { Some(&self.pelvis_rotation) } else { None }
    }

    pub fn translation_opt(&self) -> Option<&[f32]> {
        if self.global_translation.iter().any(|&v| v != 0.0) { Some(&self.global_translation) } else { None }
    }
}

pub fn load_case(idx: usize) -> CaseData {
    let case_dir = reference_data_dir().join(idx.to_string());
    CaseData {
        shape: load_f32_bin(&case_dir.join("shape.bin")),
        body_pose: load_f32_bin(&case_dir.join("body_pose.bin")),
        pelvis_rotation: load_f32_bin(&case_dir.join("pelvis_rotation.bin")),
        global_translation: load_f32_bin(&case_dir.join("global_translation.bin")),
    }
}

/// Concatenated inputs for all reference cases, suitable for batched calls.
pub struct BatchedInputs {
    pub shape: Vec<f32>,
    pub body_pose: Vec<f32>,
    pub pelvis: Vec<f32>,
    pub translation: Vec<f32>,
}

/// Load all reference cases and concatenate their inputs.
pub fn load_batched_inputs() -> BatchedInputs {
    let mut out = BatchedInputs {
        shape: Vec::new(),
        body_pose: Vec::new(),
        pelvis: Vec::new(),
        translation: Vec::new(),
    };
    for idx in 0..NUM_CASES {
        let c = load_case(idx);
        out.shape.extend_from_slice(&c.shape);
        out.body_pose.extend_from_slice(&c.body_pose);
        out.pelvis.extend_from_slice(&c.pelvis_rotation);
        out.translation.extend_from_slice(&c.global_translation);
    }
    out
}
