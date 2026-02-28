//! GPU backend tests: verify GPU output matches CPU (and thus Python).
//!
//! Requires `gpu` feature and a GPU adapter (skips gracefully if unavailable).

#![cfg(feature = "gpu")]

mod common;

use common::*;
use body_models_wasm::gpu_forward::GpuSmplModel;

#[test]
fn test_gpu_forward_vertices_matches_cpu() {
    let model = load_model();
    let gpu_model = pollster::block_on(GpuSmplModel::new(model.data));
    let cpu_model = load_model();

    for idx in 0..NUM_CASES {
        eprintln!("GPU Case {}:", idx);
        let c = load_case(idx);

        let cpu_verts = cpu_model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        );
        let gpu_verts = gpu_model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        );
        assert_allclose(&gpu_verts, &cpu_verts, &format!("case_{}_gpu_vs_cpu", idx));
    }
}

#[test]
fn test_gpu_forward_vertices_batch() {
    let model = load_model();
    let gpu_model = pollster::block_on(GpuSmplModel::new(model.data));
    let cpu_model = load_model();
    let batch = load_batched_inputs();

    // Collect per-instance CPU references
    let mut ref_vertices = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx);
        ref_vertices.extend_from_slice(&cpu_model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        ));
    }

    let gpu_batch_verts = gpu_model.forward_vertices(
        &batch.shape, &batch.body_pose, Some(&batch.pelvis), None, Some(&batch.translation), false,
    );
    assert_allclose(&gpu_batch_verts, &ref_vertices, "gpu_batch_5_vs_cpu");
}

#[test]
fn test_gpu_forward_vertices_ground_plane() {
    let model = load_model();
    let gpu_model = pollster::block_on(GpuSmplModel::new(model.data));

    for idx in 0..NUM_CASES {
        eprintln!("GPU Case {} (ground plane):", idx);
        let c = load_case(idx);
        let ref_vertices_gp = load_f32_bin(&reference_data_dir().join(idx.to_string()).join("vertices_gp.bin"));

        let gpu_verts = gpu_model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), true,
        );
        assert_allclose(&gpu_verts, &ref_vertices_gp, &format!("case_{}_gpu_gp", idx));
    }
}
