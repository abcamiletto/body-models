//! Integration tests for CPU forward pass.
//! Uses deterministic synthetic inputs and shared model asset paths.

mod common;

use common::*;

#[test]
fn test_forward_vertices_batch_equivalence() {
    let Some(model) = load_model() else {
        eprintln!("Skipping CPU test: model asset not found");
        return;
    };
    let bp_len = model.body_pose_len();
    let batch = load_batched_inputs(bp_len);

    let mut ref_vertices = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx, bp_len);
        ref_vertices.extend_from_slice(&model.forward_vertices(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            false,
        ));
    }

    let batch_verts = model.forward_vertices(
        &batch.shape,
        &batch.body_pose,
        Some(&batch.pelvis),
        None,
        Some(&batch.translation),
        false,
    );
    assert_allclose(&batch_verts, &ref_vertices, "batch_vertices");
}

#[test]
fn test_forward_skeleton_batch_equivalence() {
    let Some(model) = load_model() else {
        eprintln!("Skipping CPU test: model asset not found");
        return;
    };
    let bp_len = model.body_pose_len();
    let batch = load_batched_inputs(bp_len);

    let mut ref_transforms = Vec::new();
    for idx in 0..NUM_CASES {
        let c = load_case(idx, bp_len);
        ref_transforms.extend_from_slice(&model.forward_skeleton(
            &c.shape,
            &c.body_pose,
            c.pelvis_opt(),
            None,
            c.translation_opt(),
            false,
        ));
    }

    let batch_transforms = model.forward_skeleton(
        &batch.shape,
        &batch.body_pose,
        Some(&batch.pelvis),
        None,
        Some(&batch.translation),
        false,
    );
    assert_allclose(&batch_transforms, &ref_transforms, "batch_skeleton");
}

#[test]
fn test_ground_plane_vertex_shift() {
    let Some(model) = load_model() else {
        eprintln!("Skipping CPU test: model asset not found");
        return;
    };
    let bp_len = model.body_pose_len();
    let c = load_case(0, bp_len);

    let no_gp = model.forward_vertices(&c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), false);
    let with_gp = model.forward_vertices(&c.shape, &c.body_pose, c.pelvis_opt(), None, c.translation_opt(), true);

    for vi in 0..model.data.num_vertices {
        let dx = (with_gp[vi * 3] - no_gp[vi * 3]).abs();
        let dy = with_gp[vi * 3 + 1] - no_gp[vi * 3 + 1];
        let dz = (with_gp[vi * 3 + 2] - no_gp[vi * 3 + 2]).abs();
        assert!(dx <= 1e-4, "unexpected x shift at vertex {}", vi);
        assert!(dz <= 1e-4, "unexpected z shift at vertex {}", vi);
        assert!((dy - model.data.rest_pose_y_offset).abs() <= 1e-3, "unexpected y shift at vertex {}", vi);
    }
}

#[test]
fn test_model_loading() {
    let Some(model) = load_model() else {
        eprintln!("Skipping CPU test: model asset not found");
        return;
    };
    assert_eq!(model.data.num_vertices, 6890);
    assert_eq!(model.data.num_joints, 24);
    assert_eq!(model.data.num_pose_params, 207);
    assert_eq!(model.data.nnz_per_vertex, 4);
    assert_eq!(model.data.num_faces, 13776);
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
}
