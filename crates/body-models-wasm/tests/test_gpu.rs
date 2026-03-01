//! GPU backend tests: compare GPU output against CPU output.
//!
//! Requires `gpu` feature and a GPU adapter/device.
//! Tests are skipped gracefully when unavailable.

#![cfg(feature = "gpu")]

mod common;

use body_models_wasm::gpu_forward::GpuSmplModel;

use common::*;

fn new_gpu_or_skip() -> Option<GpuSmplModel> {
    let Some(model) = load_model() else {
        eprintln!("Skipping GPU tests: model asset not found");
        return None;
    };
    match pollster::block_on(GpuSmplModel::try_new(model.data, 64)) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("Skipping GPU tests: {}", e);
            None
        }
    }
}

#[test]
fn test_gpu_forward_vertices_matches_cpu() {
    let Some(gpu_model) = new_gpu_or_skip() else { return; };
    let Some(cpu_model) = load_model() else {
        eprintln!("Skipping GPU tests: model asset not found");
        return;
    };
    let bp_len = cpu_model.body_pose_len();

    for idx in 0..NUM_CASES {
        let c = load_case(idx, bp_len);
        let cpu_verts = cpu_model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            false,
        );
        let gpu_verts = gpu_model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            false,
        );
        assert_allclose(&gpu_verts, &cpu_verts, &format!("case_{}_gpu_vs_cpu", idx));
    }
}

#[test]
fn test_gpu_forward_vertices_batch() {
    let Some(gpu_model) = new_gpu_or_skip() else { return; };
    let Some(cpu_model) = load_model() else {
        eprintln!("Skipping GPU tests: model asset not found");
        return;
    };
    let bp_len = cpu_model.body_pose_len();
    let batch = load_batched_inputs(bp_len);

    let mut ref_vertices = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx, bp_len);
        ref_vertices.extend_from_slice(&cpu_model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            false,
        ));
    }

    let gpu_batch_verts = gpu_model.forward_vertices(
        &batch.shape,
        &batch.body_pose,
        Some(&batch.pelvis),
        None,
        Some(&batch.translation),
        false,
    );
    assert_allclose(&gpu_batch_verts, &ref_vertices, "gpu_batch_vs_cpu");
}

#[test]
fn test_gpu_forward_vertices_ground_plane() {
    let Some(gpu_model) = new_gpu_or_skip() else { return; };
    let Some(cpu_model) = load_model() else {
        eprintln!("Skipping GPU tests: model asset not found");
        return;
    };
    let bp_len = cpu_model.body_pose_len();

    for idx in 0..NUM_CASES {
        let c = load_case(idx, bp_len);
        let cpu_verts = cpu_model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            true,
        );
        let gpu_verts = gpu_model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            true,
        );
        assert_allclose(&gpu_verts, &cpu_verts, &format!("case_{}_gpu_gp_vs_cpu", idx));
    }
}

