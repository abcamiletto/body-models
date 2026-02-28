//! Integration tests for CPU forward pass.
//! Verifies Rust output matches Python NumPy reference within rtol=1e-4, atol=1e-4.

mod common;

use common::*;

#[test]
fn test_forward_vertices_all_cases() {
    let model = load_model();

    for idx in 0..NUM_CASES {
        eprintln!("Case {}:", idx);
        let c = load_case(idx);
        let ref_vertices = load_f32_bin(&reference_data_dir().join(idx.to_string()).join("vertices.bin"));

        let vertices = model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        );
        assert_allclose(&vertices, &ref_vertices, &format!("case_{}_vertices", idx));
    }
}

#[test]
fn test_forward_vertices_ground_plane() {
    let model = load_model();

    for idx in 0..NUM_CASES {
        eprintln!("Case {} (ground plane):", idx);
        let c = load_case(idx);
        let ref_vertices_gp = load_f32_bin(&reference_data_dir().join(idx.to_string()).join("vertices_gp.bin"));

        let vertices = model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), true,
        );
        assert_allclose(&vertices, &ref_vertices_gp, &format!("case_{}_vertices_gp", idx));
    }
}

#[test]
fn test_forward_skeleton_all_cases() {
    let model = load_model();

    for idx in 0..NUM_CASES {
        eprintln!("Case {} (skeleton):", idx);
        let c = load_case(idx);
        let ref_transforms = load_f32_bin(&reference_data_dir().join(idx.to_string()).join("transforms.bin"));

        let transforms = model.forward_skeleton(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        );
        assert_allclose(&transforms, &ref_transforms, &format!("case_{}_skeleton", idx));
    }
}

#[test]
fn test_forward_vertices_batch() {
    let model = load_model();
    let batch = load_batched_inputs();

    // Collect per-instance references
    let mut ref_vertices = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx);
        ref_vertices.extend_from_slice(&model.forward_vertices(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        ));
    }

    // Batched call (passing zeros for pelvis/translation is equivalent to None)
    let batch_verts = model.forward_vertices(
        &batch.shape, &batch.body_pose, Some(&batch.pelvis), None, Some(&batch.translation), false,
    );
    assert_allclose(&batch_verts, &ref_vertices, "batch_5_vertices");
}

#[test]
fn test_forward_skeleton_batch() {
    let model = load_model();
    let batch = load_batched_inputs();

    let mut ref_transforms = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx);
        ref_transforms.extend_from_slice(&model.forward_skeleton(
            &c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false,
        ));
    }

    let batch_transforms = model.forward_skeleton(
        &batch.shape, &batch.body_pose, Some(&batch.pelvis), None, Some(&batch.translation), false,
    );
    assert_allclose(&batch_transforms, &ref_transforms, "batch_5_skeleton");
}

#[test]
fn test_model_loading() {
    let model = load_model();
    assert_eq!(model.data.num_vertices, 6890);
    assert_eq!(model.data.num_joints, 24);
    assert_eq!(model.data.num_shape_params, 300);
    assert_eq!(model.data.num_pose_params, 207);
    assert_eq!(model.data.nnz_per_vertex, 4);
    assert_eq!(model.data.num_faces, 13776);
    assert!(model.data.rest_pose_y_offset > 0.0);
}

#[test]
fn test_npz_loading_directly() {
    let path = model_npz_path();
    if !path.exists() {
        eprintln!("Skipping npz test: {} not found", path.display());
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let model = body_models_wasm::SmplModel::from_bytes(&bytes).unwrap();
    assert_eq!(model.data.num_vertices, 6890);
    assert_eq!(model.data.num_joints, 24);
    assert_eq!(model.data.num_shape_params, 300);
    assert_eq!(model.data.num_pose_params, 207);
    assert_eq!(model.data.num_faces, 13776);
    eprintln!("  npz loading ✓ (V={}, J={}, S={})", model.data.num_vertices, model.data.num_joints, model.data.num_shape_params);
}

#[test]
fn test_npz_matches_bin() {
    let npz_path = model_npz_path();
    let bin_path = model_bin_path();
    if !npz_path.exists() || !bin_path.exists() {
        eprintln!("Skipping npz-vs-bin test: need both {} and {}", npz_path.display(), bin_path.display());
        return;
    }

    let npz_model = body_models_wasm::SmplModel::from_bytes(&std::fs::read(&npz_path).unwrap()).unwrap();
    let bin_model = body_models_wasm::SmplModel::from_bytes(&std::fs::read(&bin_path).unwrap()).unwrap();

    let c = load_case(3);
    let npz_verts = npz_model.forward_vertices(&c.shape, &c.body_pose, Some(&c.pelvis_rotation), None, None, true);
    let bin_verts = bin_model.forward_vertices(&c.shape, &c.body_pose, Some(&c.pelvis_rotation), None, None, true);

    assert_allclose(&npz_verts, &bin_verts, "npz_vs_bin_vertices");
}
