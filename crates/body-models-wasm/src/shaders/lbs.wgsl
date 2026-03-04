// Sparse Linear Blend Skinning (batched).
//
// Global rotation, translation, and y_offset are fused into joint_rt on the CPU.
// Each thread processes one vertex across the batch.
// Total threads = B * V.
// Workgroup size: 64 threads, dispatch: ceil(B*V / 64) workgroups.

struct Params {
    num_vertices: u32,
    nnz_per_vertex: u32,
    num_joints: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> v_shaped: array<f32>;          // [B*V*3]
@group(0) @binding(2) var<storage, read> sparse_indices: array<u32>;    // [V*K]
@group(0) @binding(3) var<storage, read> sparse_weights: array<f32>;    // [V*K]
@group(0) @binding(4) var<storage, read> joint_rt: array<f32>;          // [B*J*12]
@group(0) @binding(5) var<storage, read_write> v_posed: array<f32>;     // [B*V*3] output

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let V = params.num_vertices;
    let K = params.nnz_per_vertex;
    let J = params.num_joints;
    let B = params.batch_size;

    if flat_idx >= B * V {
        return;
    }

    let local_vi = flat_idx % V;
    let batch_idx = flat_idx / V;

    let v_offset = batch_idx * V * 3u + local_vi * 3u;
    let vx = v_shaped[v_offset];
    let vy = v_shaped[v_offset + 1u];
    let vz = v_shaped[v_offset + 2u];

    // Weighted blend of rotation and translation
    var wr00 = 0.0; var wr01 = 0.0; var wr02 = 0.0;
    var wr10 = 0.0; var wr11 = 0.0; var wr12 = 0.0;
    var wr20 = 0.0; var wr21 = 0.0; var wr22 = 0.0;
    var wt0 = 0.0; var wt1 = 0.0; var wt2 = 0.0;

    let jrt_offset = batch_idx * J * 12u;

    for (var ki = 0u; ki < K; ki = ki + 1u) {
        let ji = sparse_indices[local_vi * K + ki];
        let w = sparse_weights[local_vi * K + ki];
        let base = jrt_offset + ji * 12u;

        wr00 = wr00 + w * joint_rt[base];
        wr01 = wr01 + w * joint_rt[base + 1u];
        wr02 = wr02 + w * joint_rt[base + 2u];
        wr10 = wr10 + w * joint_rt[base + 3u];
        wr11 = wr11 + w * joint_rt[base + 4u];
        wr12 = wr12 + w * joint_rt[base + 5u];
        wr20 = wr20 + w * joint_rt[base + 6u];
        wr21 = wr21 + w * joint_rt[base + 7u];
        wr22 = wr22 + w * joint_rt[base + 8u];
        wt0 = wt0 + w * joint_rt[base + 9u];
        wt1 = wt1 + w * joint_rt[base + 10u];
        wt2 = wt2 + w * joint_rt[base + 11u];
    }

    // v_posed = W_R @ v_shaped + W_t
    let px = wr00 * vx + wr01 * vy + wr02 * vz + wt0;
    let py = wr10 * vx + wr11 * vy + wr12 * vz + wt1;
    let pz = wr20 * vx + wr21 * vy + wr22 * vz + wt2;

    v_posed[v_offset] = px;
    v_posed[v_offset + 1u] = py;
    v_posed[v_offset + 2u] = pz;
}
