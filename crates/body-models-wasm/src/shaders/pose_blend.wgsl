// Pose blend shapes: v_shaped += posedirs @ pose_delta
//
// Each thread processes one vertex component (batch, v, d).
// Total threads = B * V * 3.
// Workgroup size: 64 threads, dispatch: ceil(B*V*3 / 64) workgroups.

struct Params {
    num_vertices: u32,
    num_pose_params: u32,  // P = (J-1)*9
    batch_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> posedirs: array<f32>;      // [V*3*P]
@group(0) @binding(2) var<storage, read> pose_delta: array<f32>;    // [B*P]
@group(0) @binding(3) var<storage, read_write> v_shaped: array<f32>; // [B*V*3] in/out

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let V = params.num_vertices;
    let P = params.num_pose_params;
    let B = params.batch_size;

    if idx >= B * V * 3u {
        return;
    }

    let local_idx = idx % (V * 3u);
    let batch_idx = idx / (V * 3u);

    let vi = local_idx / 3u;
    let di = local_idx % 3u;

    // posedirs layout: [V, 3, P], indexed as [vi*3*P + di*P + pi]
    let base = vi * 3u * P + di * P;
    let pd_offset = batch_idx * P;

    var acc = 0.0;
    for (var pi = 0u; pi < P; pi = pi + 1u) {
        acc = acc + pose_delta[pd_offset + pi] * posedirs[base + pi];
    }

    v_shaped[idx] = v_shaped[idx] + acc;
}
