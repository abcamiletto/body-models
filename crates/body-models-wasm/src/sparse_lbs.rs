//! Sparse LBS weight precomputation.
//!
//! SMPL has exactly 4 non-zero weights per vertex (24 joints).
//! This module extracts the top-K sparse indices and weights
//! for efficient GPU dispatch.

/// Extract top-K sparse weights from dense LBS weights.
///
/// Input: dense weights [V, J] (row-major)
/// Output: (indices [V, K], weights [V, K])
///
/// Uses partial selection sort (O(J*K) per vertex) on a stack buffer
/// to avoid heap allocation per vertex.
pub fn extract_sparse_weights(
    dense_weights: &[f32],
    num_vertices: usize,
    num_joints: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(num_joints <= 64, "num_joints={} exceeds stack buffer size", num_joints);

    let mut indices = vec![0u32; num_vertices * k];
    let mut weights = vec![0.0f32; num_vertices * k];

    // Stack buffer for per-vertex (weight, joint_index) pairs
    let mut pairs = [(0.0f32, 0u32); 64];

    for vi in 0..num_vertices {
        let row = &dense_weights[vi * num_joints..(vi + 1) * num_joints];
        for (j, &w) in row.iter().enumerate() {
            pairs[j] = (w, j as u32);
        }

        // Partial selection sort: find top-K by swapping max to front
        let top_k = k.min(num_joints);
        for ki in 0..top_k {
            let mut best = ki;
            for j in ki + 1..num_joints {
                if pairs[j].0 > pairs[best].0 {
                    best = j;
                }
            }
            pairs.swap(ki, best);
        }

        // Renormalize top-K weights
        let sum: f32 = pairs[..top_k].iter().map(|(w, _)| w).sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

        for ki in 0..top_k {
            indices[vi * k + ki] = pairs[ki].1;
            weights[vi * k + ki] = pairs[ki].0 * inv_sum;
        }
    }

    (indices, weights)
}
