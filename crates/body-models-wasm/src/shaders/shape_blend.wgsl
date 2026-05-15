// Shape blend shapes: v_shaped = v_template + shapedirs @ shape
//
// Each thread processes one vertex component (batch, v, d) where d in {0,1,2}.
// Total threads = B * V * 3.
// Workgroup size: 64 threads, dispatch: ceil(B*V*3 / 64) workgroups.

struct Params {
    num_vertices: u32,
    num_shape_params: u32,  // S (number actually used per instance)
    batch_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> v_template: array<f32>;    // [V*3]
@group(0) @binding(2) var<storage, read> shapedirs: array<f32>;     // [V*3*S_full]
@group(0) @binding(3) var<storage, read> shape: array<f32>;         // [B*S]
@group(0) @binding(4) var<storage, read_write> v_shaped: array<f32>; // [B*V*3] output

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;  // flat index into B*V*3
    let V = params.num_vertices;
    let S = params.num_shape_params;
    let B = params.batch_size;

    if idx >= B * V * 3u {
        return;
    }

    let local_idx = idx % (V * 3u);  // index within single instance
    let batch_idx = idx / (V * 3u);  // which instance

    let vi = local_idx / 3u;
    let di = local_idx % 3u;

    var val = v_template[local_idx];

    // shapedirs layout: [V, 3, S_full], indexed as [vi*3*S_full + di*S_full + si]
    let S_full = arrayLength(&shapedirs) / (V * 3u);
    let base = vi * 3u * S_full + di * S_full;
    let shape_offset = batch_idx * S;

    for (var si = 0u; si < S; si = si + 1u) {
        val = val + shape[shape_offset + si] * shapedirs[base + si];
    }

    v_shaped[idx] = val;
}
