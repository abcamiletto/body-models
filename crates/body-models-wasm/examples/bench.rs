//! Benchmark SMPL forward pass with batch sizes B=1,4,16,64.
//!
//! Usage: cargo run --release --example bench -- <model_path>
//!
//! Outputs parseable timings for forward_vertices and forward_skeleton.

use std::time::Instant;

use body_models_wasm::SmplModel;

const N_RUNS: usize = 100;
const WARMUP: usize = 10;
const BATCH_SIZES: &[usize] = &[1, 4, 16, 64];

fn remove_outliers_and_mean(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    let filtered: Vec<f64> = sorted.into_iter().filter(|&v| v >= lower && v <= upper).collect();
    if filtered.is_empty() {
        values.iter().sum::<f64>() / values.len() as f64
    } else {
        filtered.iter().sum::<f64>() / filtered.len() as f64
    }
}

const BENCH_SHAPE_DIM: usize = 10;

/// Generate deterministic parameters for a single instance with index `i`.
fn make_params(i: usize, body_pose_len: usize) -> (Vec<f32>, Vec<f32>, [f32; 3]) {
    let offset = i as f32 * 0.7;
    let shape: Vec<f32> = (0..BENCH_SHAPE_DIM).map(|s| ((s as f32 + offset) * 0.31415).sin()).collect();
    let body_pose: Vec<f32> = (0..body_pose_len).map(|p| ((p as f32 + offset) * 0.1234).sin() * 0.3).collect();
    let pelvis = [0.1 + offset * 0.01, -0.2 + offset * 0.01, 0.15];
    (shape, body_pose, pelvis)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: bench <model_path>");
        std::process::exit(1);
    });

    eprintln!("Loading model from {}...", path);
    let bytes = std::fs::read(&path).unwrap();
    let model = SmplModel::from_bytes(&bytes).unwrap();
    eprintln!(
        "  V={}, J={}, S={}",
        model.data.num_vertices, model.data.num_joints, model.data.num_shape_params
    );

    let bp_len = model.body_pose_len();

    println!("{:<6} {:>16} {:>16}", "B", "vertices (ms)", "skeleton (ms)");
    println!("{}", "-".repeat(42));

    for &b in BATCH_SIZES {
        // Build batched inputs by repeating diverse params
        let mut all_shape = Vec::with_capacity(b * BENCH_SHAPE_DIM);
        let mut all_body_pose = Vec::with_capacity(b * bp_len);
        let mut all_pelvis = Vec::with_capacity(b * 3);
        for i in 0..b {
            let (s, bp, pr) = make_params(i, bp_len);
            all_shape.extend_from_slice(&s);
            all_body_pose.extend_from_slice(&bp);
            all_pelvis.extend_from_slice(&pr);
        }

        // --- forward_vertices ---
        for _ in 0..WARMUP {
            let _ = model.forward_vertices(&all_shape, &all_body_pose, Some(&all_pelvis), None, None, true);
        }

        let mut times = Vec::with_capacity(N_RUNS);
        for _ in 0..N_RUNS {
            let t0 = Instant::now();
            let _ = model.forward_vertices(&all_shape, &all_body_pose, Some(&all_pelvis), None, None, true);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let verts_ms = remove_outliers_and_mean(&times);

        // --- forward_skeleton ---
        for _ in 0..WARMUP {
            let _ = model.forward_skeleton(&all_shape, &all_body_pose, Some(&all_pelvis), None, None, true);
        }

        let mut times = Vec::with_capacity(N_RUNS);
        for _ in 0..N_RUNS {
            let t0 = Instant::now();
            let _ = model.forward_skeleton(&all_shape, &all_body_pose, Some(&all_pelvis), None, None, true);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let skel_ms = remove_outliers_and_mean(&times);

        println!("{:<6} {:>16.3} {:>16.3}", b, verts_ms, skel_ms);
    }
}
